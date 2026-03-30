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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include "paddle/extension.h"
#include "../ngram_match_common.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// ============================================================
// Phase 1 mixed search kernel — one block per batch item
// ============================================================
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

  // Skip batch items with no active tokens (matches CPU path logic)
  if (seq_lens_this_time[batch_idx] == 0) return;

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
// Phase 2 mixed gather kernel — serial threshold + copy
// ============================================================
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
    int max_draft_tokens =
        static_cast<int>(min(static_cast<int64_t>(max_draft_tokens_param -
                                                  ori_seq_len_this_time + 1),
                             max_dec_len[batch_idx] - step_idx[batch_idx] - 1));

    if (ori_seq_len_this_time == 0 || max_draft_tokens <= 0) {
      continue;
    }

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
      haystack = pre_ids + batch_idx * pre_ids_stride;
      haystack_len = step_idx[batch_idx];
    }

    int64_t start_idx = res.match_pos + res.ngram_size;
    int64_t end_idx =
        min(start_idx + static_cast<int64_t>(max_draft_tokens), haystack_len);
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

// ============================================================
// CPU path — preserved from original for backward compatibility
// with CPU-only callers and tests.
// ============================================================
static int sum_mixed_cpu(const int *value, int num) {
  int sum_value = 0;
  for (int i = 0; i <= num; i++) {
    sum_value += value[i];
  }
  return sum_value;
}

static void find_candidate_pred_tokens_mixed(const int64_t *input_ids,
                                             const int64_t *input_ids_len,
                                             const int64_t *pre_ids,
                                             const int64_t *step_idx,
                                             const int *draft_token_num,
                                             int64_t *draft_tokens,
                                             int32_t *seq_lens_this_time,
                                             int32_t *seq_lens_decoder,
                                             int64_t *max_dec_len,
                                             int64_t input_ids_stride,
                                             int64_t pre_ids_stride,
                                             int64_t draft_tokens_stride,
                                             int64_t max_batch_size,
                                             int max_ngram_size = 3,
                                             int min_ngram_size = 1,
                                             const int max_draft_tokens = 10) {
  int threshold = 1024;
  char *env_var = getenv("SPEC_TOKENUM_THRESHOLD");
  if (env_var) {
    threshold = std::stoi(env_var);
  }
  int unprocessed_batch_size = 0;
  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    if (seq_lens_decoder[batch_idx] > 0) {
      unprocessed_batch_size++;
    }
  }
  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];
    int max_draft_tokens_query = std::min(
        static_cast<int64_t>(max_draft_tokens - ori_seq_len_this_time + 1),
        max_dec_len[batch_idx] - step_idx[batch_idx] - 1);

    if (ori_seq_len_this_time == 0 || max_draft_tokens_query <= 0) {
      continue;
    }

    const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
    int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
    const int64_t *cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
    const int64_t cur_step_idx = step_idx[batch_idx];
    const int64_t cur_input_ids_len = input_ids_len[batch_idx];
    unprocessed_batch_size--;

    auto sum_token_num = sum_mixed_cpu(seq_lens_this_time, batch_idx);
    int left_min_token_num = unprocessed_batch_size;

    if (sum_token_num + max_draft_tokens_query + left_min_token_num >
        threshold) {
      int tmp_max_draft_tokens = threshold - sum_token_num - left_min_token_num;
      max_draft_tokens_query =
          std::min(max_draft_tokens_query, tmp_max_draft_tokens);
    }

    if (sum_token_num + left_min_token_num >= threshold - 1) {
      continue;
    }
    bool match_global = false;
    for (int ngram_size = max_ngram_size;
         ngram_size >= min_ngram_size && !match_global;
         --ngram_size) {
      if (cur_step_idx < ngram_size) {
        continue;
      }
      const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

      for (int64_t i = 0; i <= cur_input_ids_len - ngram_size && !match_global;
           ++i) {
        bool match_local = true;
        for (int j = 0; j < ngram_size; j++) {
          if (ngram[j] != cur_input_ids[i + j]) {
            match_local = false;
            break;
          }
        }
        if (match_local) {
          int64_t start_idx = i + ngram_size;
          int64_t end_idx =
              std::min(start_idx + max_draft_tokens_query, cur_input_ids_len);
          if (start_idx >= end_idx) continue;

          int64_t cur_draft_token_num = end_idx - start_idx;
          seq_lens_this_time[batch_idx] =
              ori_seq_len_this_time + cur_draft_token_num;
          memcpy(cur_draft_tokens + ori_seq_len_this_time,
                 cur_input_ids + start_idx,
                 sizeof(int64_t) * cur_draft_token_num);
          match_global = true;
          break;
        }
      }
      if (!match_global) {
        for (int64_t i = 0; i <= cur_step_idx - ngram_size && !match_global;
             ++i) {
          bool match_local = true;
          for (int j = 0; j < ngram_size; j++) {
            if (ngram[j] != cur_pre_ids[i + j]) {
              match_local = false;
              break;
            }
          }
          if (match_local) {
            int64_t start_idx = i + ngram_size;
            int64_t end_idx =
                std::min(start_idx + max_draft_tokens_query, cur_step_idx);
            int64_t cur_draft_token_num = end_idx - start_idx;
            if (start_idx >= end_idx) continue;

            seq_lens_this_time[batch_idx] =
                ori_seq_len_this_time + cur_draft_token_num;
            memcpy(cur_draft_tokens + ori_seq_len_this_time,
                   cur_pre_ids + start_idx,
                   sizeof(int64_t) * cur_draft_token_num);
            match_global = true;
            break;
          }
        }
      }
    }
  }
}

// ============================================================
// GPU path — Two-phase parallel CUDA kernels for hybrid ngram matching.
//
// Phase 1: <<<bsz, NGRAM_BLOCK_THREADS>>> — parallel sliding-window
//          search within each batch item (256 threads per batch).
// Phase 2: <<<1, 1>>> — serial threshold + token copy (inter-batch
//          dependency via running sum of seq_lens_this_time).
// ============================================================

void HybridMtpNgram(const paddle::Tensor &input_ids,
                    const paddle::Tensor &input_ids_len,
                    const paddle::Tensor &pre_ids,
                    const paddle::Tensor &step_idx,
                    const paddle::Tensor &draft_token_num,
                    const paddle::Tensor &draft_tokens,
                    const paddle::Tensor &seq_lens_this_time,
                    const paddle::Tensor &seq_lens_decoder,
                    const paddle::Tensor &max_dec_len,
                    const int max_ngram_size,
                    const int min_ngram_size,
                    const int max_draft_tokens) {
  auto input_ids_shape = input_ids.shape();
  const int64_t input_ids_stride = input_ids_shape[1];

  auto pre_ids_shape = pre_ids.shape();
  const int64_t pre_ids_stride = pre_ids_shape[1];

  auto draft_tokens_shape = draft_tokens.shape();
  const int64_t draft_tokens_stride = draft_tokens_shape[1];

  const int64_t max_batch_size = seq_lens_this_time.shape()[0];

  int threshold = 1024;
  const char *env_var = getenv("SPEC_TOKENUM_THRESHOLD");
  if (env_var) {
    threshold = std::stoi(env_var);
  }

  if (input_ids.is_gpu()) {
    auto stream = input_ids.stream();

    // Allocate scratch buffer for Phase 1 → Phase 2 communication
    auto match_buf = paddle::empty(
        {max_batch_size * static_cast<int64_t>(sizeof(NgramMatchResult))},
        paddle::DataType::UINT8,
        input_ids.place());
    auto *match_results =
        reinterpret_cast<NgramMatchResult *>(match_buf.data<uint8_t>());

    // Phase 1: parallel search — one block per batch, 256 threads per block
    ngram_match_mixed_search_kernel<<<max_batch_size,
                                      NGRAM_BLOCK_THREADS,
                                      0,
                                      stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        seq_lens_this_time.data<int32_t>(),
        input_ids_stride,
        pre_ids_stride,
        max_batch_size,
        max_ngram_size,
        min_ngram_size,
        match_results);

    // Phase 2: serial threshold + token copy (same stream = ordered)
    ngram_match_mixed_gather_kernel<<<1, 1, 0, stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
        seq_lens_decoder.data<int32_t>(),
        max_dec_len.data<int64_t>(),
        input_ids_stride,
        pre_ids_stride,
        draft_tokens_stride,
        max_batch_size,
        max_draft_tokens,
        threshold,
        match_results);
  } else {
    find_candidate_pred_tokens_mixed(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
        const_cast<int32_t *>(seq_lens_decoder.data<int32_t>()),
        const_cast<int64_t *>(max_dec_len.data<int64_t>()),
        input_ids_stride,
        pre_ids_stride,
        draft_tokens_stride,
        max_batch_size,
        max_ngram_size,
        min_ngram_size,
        max_draft_tokens);
  }
}

PD_BUILD_STATIC_OP(hybrid_mtp_ngram)
    .Inputs({"input_ids",
             "input_ids_len",
             "pre_ids",
             "step_idx",
             "draft_token_num",
             "draft_tokens",
             "seq_lens_this_time",
             "seq_lens_decoder",
             "max_dec_len"})
    .Attrs({"max_ngram_size: int",
            "min_ngram_size: int",
            "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
    .SetKernelFn(PD_KERNEL(HybridMtpNgram))
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}});
