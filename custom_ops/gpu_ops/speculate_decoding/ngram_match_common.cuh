// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <climits>

// Shared ngram matching logic used by both ngram_match_kernel and
// ngram_match_mixed_kernel.  Extracted per upstream requirement:
// "两个Kernel逻辑有较为相似部分，Kernel 形式为提取共用的匹配逻辑，外加业务逻辑"
//
// Two-phase parallel architecture:
//   Phase 1 — <<<bsz, NGRAM_BLOCK_THREADS>>>: parallel sliding-window search
//   Phase 2 — <<<1, 1>>>: serial threshold + token copy (inter-batch dep)

#define NGRAM_BLOCK_THREADS 256

// Intermediate result for one batch item produced by Phase 1 (parallel search)
// and consumed by Phase 2 (serial threshold + copy).
struct NgramMatchResult {
  int64_t match_pos;  // first (leftmost) match position in haystack (-1=none)
  int ngram_size;     // which ngram_size produced this match
  int haystack_type;  // 0 = input_ids, 1 = pre_ids
};

// ------------------------------------------------------------
// atomicMin for int64_t via CAS loop.  CUDA has no native
// int64 atomicMin.  All values are non-negative positions or
// INT64_MAX, so unsigned reinterpretation is safe.
// ------------------------------------------------------------
__device__ __forceinline__ void atomicMin64(int64_t *addr, int64_t val) {
  unsigned long long *addr_ull = reinterpret_cast<unsigned long long *>(addr);
  unsigned long long val_ull = static_cast<unsigned long long>(val);
  unsigned long long old = *addr_ull;
  while (val_ull < old) {
    unsigned long long assumed = old;
    old = atomicCAS(addr_ull, assumed, val_ull);
    if (old == assumed) break;
  }
}

// ------------------------------------------------------------
// parallel_ngram_search — Block-cooperative haystack search.
//
// Called by NGRAM_BLOCK_THREADS threads within a single block.
// Searches for ngram[0..ngram_size-1] in haystack[0..haystack_len-1].
// Uses shared-memory s_min_pos to reduce to the FIRST (leftmost)
// match position.
//
// Returns the leftmost match position, or INT64_MAX if no match.
// Caller must provide __shared__ int64_t s_min_pos.
// ------------------------------------------------------------
__device__ __forceinline__ int64_t
parallel_ngram_search(const int64_t *haystack,
                      int64_t haystack_len,
                      const int64_t *ngram,
                      int ngram_size,
                      int64_t *s_min_pos) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  if (tid == 0) {
    *s_min_pos = INT64_MAX;
  }
  __syncthreads();

  int64_t search_len = haystack_len - ngram_size + 1;
  if (search_len <= 0) {
    __syncthreads();
    return *s_min_pos;
  }

  for (int64_t i = tid; i < search_len; i += nthreads) {
    bool match = true;
    for (int j = 0; j < ngram_size; j++) {
      if (ngram[j] != haystack[i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      atomicMin64(s_min_pos, i);
    }
  }
  __syncthreads();

  return *s_min_pos;
}

// ============================================================
// Phase 1 search kernels — one block per batch item
// ============================================================

// ngram_match Phase 1: parallel search across all batch items.
// Each block processes one batch item with NGRAM_BLOCK_THREADS threads.
__global__ void ngram_match_search_kernel(const int64_t *input_ids,
                                          const int64_t *input_ids_len,
                                          const int64_t *token_ids_all,
                                          const int64_t *prompt_lens,
                                          const int64_t *step_idx,
                                          const int32_t *seq_lens_encoder,
                                          const int32_t *seq_lens_decoder,
                                          int64_t input_ids_stride,
                                          int64_t max_model_len,
                                          int64_t max_batch_size,
                                          int max_ngram_size,
                                          NgramMatchResult *match_results) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= max_batch_size) return;

  __shared__ int64_t s_min_pos;

  if (threadIdx.x == 0) {
    match_results[batch_idx].match_pos = -1;
    match_results[batch_idx].ngram_size = 0;
    match_results[batch_idx].haystack_type = 0;
  }
  __syncthreads();

  if (seq_lens_encoder[batch_idx] > 0) return;
  if (seq_lens_decoder[batch_idx] == 0) return;

  const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
  const int64_t cur_input_ids_len = input_ids_len[batch_idx];
  const int64_t prompt_len = prompt_lens[batch_idx];
  const int64_t *cur_pre_ids =
      token_ids_all + batch_idx * max_model_len + prompt_len;
  const int64_t cur_step_idx = step_idx[batch_idx];

  for (int ngram_size = max_ngram_size; ngram_size >= 1; --ngram_size) {
    if (cur_step_idx < ngram_size) continue;

    const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

    int64_t pos = parallel_ngram_search(
        cur_input_ids, cur_input_ids_len, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      if (threadIdx.x == 0) {
        match_results[batch_idx].match_pos = pos;
        match_results[batch_idx].ngram_size = ngram_size;
        match_results[batch_idx].haystack_type = 0;
      }
      return;
    }

    pos = parallel_ngram_search(
        cur_pre_ids, cur_step_idx, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      if (threadIdx.x == 0) {
        match_results[batch_idx].match_pos = pos;
        match_results[batch_idx].ngram_size = ngram_size;
        match_results[batch_idx].haystack_type = 1;
      }
      return;
    }
  }
}

// ngram_match_mixed Phase 1: parallel search across all batch items.
__global__ void ngram_match_mixed_search_kernel(
    const int64_t *input_ids,
    const int64_t *input_ids_len,
    const int64_t *pre_ids,
    const int64_t *step_idx,
    const int32_t *seq_lens_this_time,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t max_batch_size,
    int max_ngram_size,
    int min_ngram_size,
    NgramMatchResult *match_results) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= max_batch_size) return;

  __shared__ int64_t s_min_pos;

  if (threadIdx.x == 0) {
    match_results[batch_idx].match_pos = -1;
    match_results[batch_idx].ngram_size = 0;
    match_results[batch_idx].haystack_type = 0;
  }
  __syncthreads();

  const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];
  if (ori_seq_len_this_time == 0) return;

  const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
  const int64_t cur_input_ids_len = input_ids_len[batch_idx];
  const int64_t *cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
  const int64_t cur_step_idx = step_idx[batch_idx];

  for (int ngram_size = max_ngram_size; ngram_size >= min_ngram_size;
       --ngram_size) {
    if (cur_step_idx < ngram_size) continue;

    const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

    int64_t pos = parallel_ngram_search(
        cur_input_ids, cur_input_ids_len, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      if (threadIdx.x == 0) {
        match_results[batch_idx].match_pos = pos;
        match_results[batch_idx].ngram_size = ngram_size;
        match_results[batch_idx].haystack_type = 0;
      }
      return;
    }

    pos = parallel_ngram_search(
        cur_pre_ids, cur_step_idx, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      if (threadIdx.x == 0) {
        match_results[batch_idx].match_pos = pos;
        match_results[batch_idx].ngram_size = ngram_size;
        match_results[batch_idx].haystack_type = 1;
      }
      return;
    }
  }
}

// ============================================================
// Phase 2 gather kernels — serial threshold + copy (<<<1,1>>>)
// ============================================================

// ngram_match Phase 2: serial threshold + token copy.
__global__ void ngram_match_gather_kernel(
    const int64_t *input_ids,
    const int64_t *input_ids_len,
    const int64_t *token_ids_all,
    const int64_t *prompt_lens,
    const int64_t *step_idx,
    const int *draft_token_num,
    int64_t *draft_tokens,
    int32_t *seq_lens_this_time,
    const int32_t *seq_lens_encoder,
    const int32_t *seq_lens_decoder,
    const int64_t *max_dec_len,
    int64_t input_ids_stride,
    int64_t max_model_len,
    int64_t draft_tokens_stride,
    int64_t max_batch_size,
    int max_draft_tokens_param,
    int threshold,
    const NgramMatchResult *match_results) {
  int unprocessed_batch_size = 0;
  for (int i = 0; i < max_batch_size; i++) {
    if (seq_lens_encoder[i] > 0 || seq_lens_decoder[i] > 0) {
      unprocessed_batch_size++;
    }
  }

  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    int64_t remaining = max_dec_len[batch_idx] - step_idx[batch_idx] - 1;
    int max_draft_tokens = static_cast<int>(
        min(static_cast<int64_t>(draft_token_num[batch_idx]), remaining));

    if (seq_lens_encoder[batch_idx] > 0) {
      continue;
    } else if (seq_lens_decoder[batch_idx] == 0) {
      seq_lens_this_time[batch_idx] = 0;
      continue;
    }

    seq_lens_this_time[batch_idx] = 1;
    unprocessed_batch_size--;

    int sum_token_num = 0;
    for (int i = 0; i <= batch_idx; i++) {
      sum_token_num += seq_lens_this_time[i];
    }
    int left_min_token_num = unprocessed_batch_size;

    if (sum_token_num + max_draft_tokens + left_min_token_num > threshold) {
      int tmp = threshold - sum_token_num - left_min_token_num;
      max_draft_tokens = min(tmp, max_draft_tokens);
    }

    if (sum_token_num + left_min_token_num >= threshold - 1) {
      continue;
    }

    const NgramMatchResult &res = match_results[batch_idx];
    if (res.match_pos < 0) continue;

    const int64_t *haystack;
    int64_t haystack_len;
    if (res.haystack_type == 0) {
      haystack = input_ids + batch_idx * input_ids_stride;
      haystack_len = input_ids_len[batch_idx];
    } else {
      int64_t pl = prompt_lens[batch_idx];
      haystack = token_ids_all + batch_idx * max_model_len + pl;
      haystack_len = step_idx[batch_idx];
    }

    int64_t start_idx = res.match_pos + res.ngram_size;
    int64_t end_idx =
        min(start_idx + static_cast<int64_t>(max_draft_tokens), haystack_len);
    if (start_idx >= end_idx) continue;

    int64_t n = end_idx - start_idx;
    seq_lens_this_time[batch_idx] = static_cast<int32_t>(1 + n);
    int64_t *cur_draft = draft_tokens + batch_idx * draft_tokens_stride;
    for (int64_t k = 0; k < n; k++) {
      cur_draft[1 + k] = haystack[start_idx + k];
    }
  }
}

// ngram_match_mixed Phase 2: serial threshold + token copy.
__global__ void ngram_match_mixed_gather_kernel(
    const int64_t *input_ids,
    const int64_t *input_ids_len,
    const int64_t *pre_ids,
    const int64_t *step_idx,
    const int *draft_token_num,
    int64_t *draft_tokens,
    int32_t *seq_lens_this_time,
    const int32_t *seq_lens_decoder,
    const int64_t *max_dec_len,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t draft_tokens_stride,
    int64_t max_batch_size,
    int max_draft_tokens_param,
    int threshold,
    const NgramMatchResult *match_results) {
  int unprocessed_batch_size = 0;
  for (int i = 0; i < max_batch_size; i++) {
    if (seq_lens_decoder[i] > 0) {
      unprocessed_batch_size++;
    }
  }

  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];
    int64_t remaining = max_dec_len[batch_idx] - step_idx[batch_idx] - 1;
    int64_t max_query_64 = min(static_cast<int64_t>(max_draft_tokens_param -
                                                    ori_seq_len_this_time + 1),
                               remaining);
    int max_draft_tokens_query = static_cast<int>(max_query_64);

    if (ori_seq_len_this_time == 0 || max_draft_tokens_query <= 0) {
      continue;
    }

    unprocessed_batch_size--;

    int sum_token_num = 0;
    for (int i = 0; i <= batch_idx; i++) {
      sum_token_num += seq_lens_this_time[i];
    }
    int left_min_token_num = unprocessed_batch_size;

    if (sum_token_num + max_draft_tokens_query + left_min_token_num >
        threshold) {
      int tmp = threshold - sum_token_num - left_min_token_num;
      max_draft_tokens_query = min(max_draft_tokens_query, tmp);
    }

    if (sum_token_num + left_min_token_num >= threshold - 1) {
      continue;
    }

    const NgramMatchResult &res = match_results[batch_idx];
    if (res.match_pos < 0) continue;

    const int64_t *haystack;
    int64_t haystack_len;
    if (res.haystack_type == 0) {
      haystack = input_ids + batch_idx * input_ids_stride;
      haystack_len = input_ids_len[batch_idx];
    } else {
      haystack = pre_ids + batch_idx * pre_ids_stride;
      haystack_len = step_idx[batch_idx];
    }

    int64_t start_idx = res.match_pos + res.ngram_size;
    int64_t end_idx = min(
        start_idx + static_cast<int64_t>(max_draft_tokens_query), haystack_len);
    if (start_idx >= end_idx) continue;

    int64_t n = end_idx - start_idx;
    seq_lens_this_time[batch_idx] =
        static_cast<int32_t>(ori_seq_len_this_time + n);
    int64_t *cur_draft = draft_tokens + batch_idx * draft_tokens_stride;
    for (int64_t k = 0; k < n; k++) {
      cur_draft[ori_seq_len_this_time + k] = haystack[start_idx + k];
    }
  }
}
