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
#include <cub/cub.cuh>
#include "paddle/extension.h"
#include "ngram_match_common.cuh"

// ============================================================
// Phase 1 search kernel — one block per batch item.
// Finds the leftmost ngram match and writes tentative draft
// tokens to a scratch buffer (draft_tokens_copy) along with
// the tentative new seq_lens_this_time to a copy buffer.
// Phase 2 will decide which ones to keep (threshold logic).
// ============================================================
__global__ void ngram_match_search_kernel(const int64_t *input_ids,
                                          const int64_t *input_ids_len,
                                          const int64_t *token_ids_all,
                                          const int64_t *prompt_lens,
                                          const int64_t *step_idx,
                                          const int *draft_token_num,
                                          const int32_t *seq_lens_encoder,
                                          const int32_t *seq_lens_decoder,
                                          const int64_t *max_dec_len,
                                          int64_t *draft_tokens_copy,
                                          int32_t *seq_lens_this_time_copy,
                                          int64_t input_ids_stride,
                                          int64_t max_model_len,
                                          int64_t draft_tokens_stride,
                                          int64_t max_batch_size,
                                          int max_ngram_size) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= max_batch_size) return;

  __shared__ int64_t s_min_pos;

  if (threadIdx.x == 0) {
    // Default: 0 = this item contributes nothing to threshold budget.
    // Active decoder items will be set to 1+ below.
    seq_lens_this_time_copy[batch_idx] = 0;
  }
  __syncthreads();

  // Skip if encoder active (preserves original seq_lens_this_time) or
  // decoder inactive (Phase 2 writes 0 for these).
  if (seq_lens_encoder[batch_idx] > 0) return;
  if (seq_lens_decoder[batch_idx] == 0) return;

  // Active decoder item: at least the base token.
  if (threadIdx.x == 0) seq_lens_this_time_copy[batch_idx] = 1;

  const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
  const int64_t cur_input_ids_len = input_ids_len[batch_idx];
  const int64_t prompt_len = prompt_lens[batch_idx];
  const int64_t *cur_pre_ids =
      token_ids_all + batch_idx * max_model_len + prompt_len;
  const int64_t cur_step_idx = step_idx[batch_idx];

  // Compute max_draft_tokens for this batch item
  int64_t remaining = max_dec_len[batch_idx] - cur_step_idx - 1;
  if (remaining <= 0) return;
  int max_draft_tokens = static_cast<int>(
      min(static_cast<int64_t>(draft_token_num[batch_idx]), remaining));

  for (int ngram_size = max_ngram_size; ngram_size >= 1; --ngram_size) {
    if (cur_step_idx < ngram_size) continue;

    const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

    int64_t pos = parallel_ngram_search(
        cur_input_ids, cur_input_ids_len, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      int64_t start_idx = pos + ngram_size;
      int64_t end_idx = min(start_idx + static_cast<int64_t>(max_draft_tokens),
                            cur_input_ids_len);
      if (threadIdx.x == 0 && start_idx < end_idx) {
        // Tentative token copy to scratch
        int64_t n = end_idx - start_idx;
        seq_lens_this_time_copy[batch_idx] = static_cast<int32_t>(1 + n);
        int64_t *dst = draft_tokens_copy + batch_idx * draft_tokens_stride;
        for (int64_t k = 0; k < n; k++) {
          dst[1 + k] = cur_input_ids[start_idx + k];
        }
      }
      // Only early-exit when tokens were actually produced
      if (start_idx < end_idx) {
        return;
      }
    }

    pos = parallel_ngram_search(
        cur_pre_ids, cur_step_idx, ngram, ngram_size, &s_min_pos);
    if (pos != INT64_MAX) {
      int64_t start_idx = pos + ngram_size;
      int64_t end_idx =
          min(start_idx + static_cast<int64_t>(max_draft_tokens), cur_step_idx);
      if (threadIdx.x == 0 && start_idx < end_idx) {
        // Tentative token copy to scratch
        int64_t n = end_idx - start_idx;
        seq_lens_this_time_copy[batch_idx] = static_cast<int32_t>(1 + n);
        int64_t *dst = draft_tokens_copy + batch_idx * draft_tokens_stride;
        for (int64_t k = 0; k < n; k++) {
          dst[1 + k] = cur_pre_ids[start_idx + k];
        }
      }
      // Only early-exit when tokens were actually produced
      if (start_idx < end_idx) {
        return;
      }
    }
  }
}

// ============================================================
// Phase 2 gather kernel — BlockScan threshold + copy
//   <<<1, NGRAM_GATHER_THREADS>>>
//
// Reads tentative allocations from Phase 1 scratch buffers,
// computes prefix sums to enforce the global threshold, then
// writes final seq_lens_this_time and copies draft tokens.
// ============================================================
__global__ void ngram_match_gather_kernel(
    const int64_t *draft_tokens_copy,
    const int32_t *seq_lens_this_time_copy,
    const int32_t *seq_lens_encoder,
    int64_t *draft_tokens,
    int32_t *seq_lens_this_time,
    int64_t draft_tokens_stride,
    int64_t max_batch_size,
    int threshold) {
  typedef cub::BlockScan<int, NGRAM_GATHER_THREADS> BlockScanInt;
  __shared__ typename BlockScanInt::TempStorage temp_storage1;
  __shared__ typename BlockScanInt::TempStorage temp_storage2;
  __shared__ int s_total_active;

  int tid = threadIdx.x;

  // Load tentative values from Phase 1.
  // Encoder-active items are included in the scan with their original
  // seq_lens_this_time to match CPU threshold-budget accounting.
  int tentative = 0;
  int is_active = 0;
  if (tid < max_batch_size) {
    if (seq_lens_encoder[tid] > 0) {
      // Encoder-active: contribute original token count to threshold budget.
      // seq_lens_this_time[tid] is still unmodified at this point.
      tentative = seq_lens_this_time[tid];
      is_active = 1;
    } else {
      tentative = seq_lens_this_time_copy[tid];
      is_active = (tentative > 0) ? 1 : 0;
    }
  }

  // Scan 1: inclusive prefix sum of tentative token counts
  int token_prefix;
  BlockScanInt(temp_storage1).InclusiveSum(tentative, token_prefix);
  __syncthreads();

  // Scan 2: inclusive prefix sum of active-item indicators
  int active_prefix;
  BlockScanInt(temp_storage2).InclusiveSum(is_active, active_prefix);
  __syncthreads();

  // Total active count from the last valid thread
  if (tid ==
      min(static_cast<int>(max_batch_size) - 1, NGRAM_GATHER_THREADS - 1)) {
    s_total_active = active_prefix;
  }
  __syncthreads();

  if (tid < max_batch_size) {
    // Encoder-active items: preserve original seq_lens_this_time.
    if (seq_lens_encoder[tid] > 0) return;

    if (tentative == 0) {
      seq_lens_this_time[tid] = 0;
      return;
    }

    int exclusive_token_prefix = token_prefix - tentative;
    int remaining_active = s_total_active - active_prefix;

    // Budget: total threshold minus tokens already allocated before me,
    // minus at-least-1 reservation for every active item after me.
    int budget = threshold - exclusive_token_prefix - remaining_active;

    int actual;
    if (budget <= 1) {
      actual = 1;  // base token only
    } else {
      actual = min(tentative, budget);
    }

    seq_lens_this_time[tid] = actual;

    // Copy draft tokens (slots 1..actual-1) from scratch to output
    if (actual > 1) {
      int64_t *dst = draft_tokens + tid * draft_tokens_stride;
      const int64_t *src = draft_tokens_copy + tid * draft_tokens_stride;
      for (int k = 1; k < actual; k++) {
        dst[k] = src[k];
      }
    }
  }
}

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// ============================================================
// CPU path — preserved from original ngram_match.cc for
// backward compatibility with CPU-only callers and tests.
// ============================================================
static int sum_cpu(const int *value, int num) {
  int sum_value = 0;
  for (int i = 0; i <= num; i++) {
    sum_value += value[i];
  }
  return sum_value;
}

static void find_candidate_pred_tokens(const int64_t *input_ids,
                                       const int64_t *input_ids_len,
                                       const int64_t *token_ids_all,
                                       const int64_t *prompt_lens,
                                       const int64_t *step_idx,
                                       const int *draft_token_num,
                                       int64_t *draft_tokens,
                                       int32_t *seq_lens_this_time,
                                       int32_t *seq_lens_encoder,
                                       int32_t *seq_lens_decoder,
                                       int64_t *max_dec_len,
                                       int64_t input_ids_stride,
                                       int64_t max_model_len,
                                       int64_t draft_tokens_stride,
                                       int64_t max_batch_size,
                                       int max_ngram_size = 3,
                                       int max_draft_tokens = 10) {
  int threshold = 128;
  char *env_var = getenv("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD");
  if (env_var) {
    threshold = std::stoi(env_var);
  }
  int unprocessed_batch_size = 0;
  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    if (seq_lens_encoder[batch_idx] > 0 || seq_lens_decoder[batch_idx] > 0) {
      unprocessed_batch_size++;
    }
  }
  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    max_draft_tokens =
        std::min(static_cast<int64_t>(draft_token_num[batch_idx]),
                 max_dec_len[batch_idx] - step_idx[batch_idx] - 1);
    if (seq_lens_encoder[batch_idx] > 0) {
      continue;
    } else if (seq_lens_decoder[batch_idx] == 0) {
      seq_lens_this_time[batch_idx] = 0;
      continue;
    }

    const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
    int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
    const int64_t *cur_pre_ids =
        token_ids_all + batch_idx * max_model_len + prompt_lens[batch_idx];
    const int64_t cur_step_idx = step_idx[batch_idx];
    const int64_t cur_input_ids_len = input_ids_len[batch_idx];
    seq_lens_this_time[batch_idx] = 1;
    unprocessed_batch_size--;

    auto sum_token_num = sum_cpu(seq_lens_this_time, batch_idx);
    int left_min_token_num = unprocessed_batch_size;

    if (sum_token_num + max_draft_tokens + left_min_token_num > threshold) {
      int tmp_max_draft_tokens = threshold - sum_token_num - left_min_token_num;
      max_draft_tokens = tmp_max_draft_tokens < max_draft_tokens
                             ? tmp_max_draft_tokens
                             : max_draft_tokens;
    }

    if (sum_token_num + left_min_token_num >= threshold - 1) {
      continue;
    }

    for (int ngram_size = max_ngram_size; ngram_size > 0; --ngram_size) {
      if (cur_step_idx < ngram_size) {
        continue;
      }
      const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

      bool match_input = false;
      for (int64_t i = 0; i <= cur_input_ids_len - ngram_size; ++i) {
        bool match = true;
        for (int j = 0; j < ngram_size; j++) {
          if (ngram[j] != cur_input_ids[i + j]) {
            match = false;
            break;
          }
        }
        if (match) {
          int64_t start_idx = i + ngram_size;
          int64_t end_idx =
              std::min(start_idx + max_draft_tokens, cur_input_ids_len);
          if (start_idx >= end_idx) continue;

          int64_t cur_draft_token_num = end_idx - start_idx;
          seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
          memcpy(cur_draft_tokens + 1,
                 cur_input_ids + start_idx,
                 sizeof(int64_t) * cur_draft_token_num);
          ngram_size = 0;
          match_input = true;
          break;
        }
      }
      if (!match_input) {
        for (int64_t i = 0; i <= cur_step_idx - ngram_size; ++i) {
          bool match = true;
          for (int j = 0; j < ngram_size; j++) {
            if (ngram[j] != cur_pre_ids[i + j]) {
              match = false;
              break;
            }
          }
          if (match) {
            int64_t start_idx = i + ngram_size;
            int64_t end_idx =
                std::min(start_idx + max_draft_tokens, cur_step_idx);
            int64_t cur_draft_token_num = end_idx - start_idx;
            if (start_idx >= end_idx) continue;

            seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
            memcpy(cur_draft_tokens + 1,
                   cur_pre_ids + start_idx,
                   sizeof(int64_t) * cur_draft_token_num);
            ngram_size = 0;
            break;
          }
        }
      }
    }
  }
}

// ============================================================
// GPU path — Two-phase parallel CUDA kernels for ngram matching.
//
// Phase 1: <<<bsz, NGRAM_BLOCK_THREADS>>> — parallel sliding-window
//          search within each batch item (NGRAM_BLOCK_THREADS threads
//          per block).  Also copies matched draft tokens to scratch.
// Phase 2: <<<1, NGRAM_GATHER_THREADS>>> — CUB BlockScan prefix-sum
//          threshold enforcement + final token copy.
//
// Phase 1 is O(bsz × seq_len × ngram_size) distributed across
// bsz × NGRAM_BLOCK_THREADS threads.  Phase 2 is O(bsz) with scans.
// ============================================================

void NgramMatch(const paddle::Tensor &input_ids,
                const paddle::Tensor &input_ids_len,
                const paddle::Tensor &token_ids_all,
                const paddle::Tensor &prompt_lens,
                const paddle::Tensor &step_idx,
                const paddle::Tensor &draft_token_num,
                const paddle::Tensor &draft_tokens,
                const paddle::Tensor &seq_lens_this_time,
                const paddle::Tensor &seq_lens_encoder,
                const paddle::Tensor &seq_lens_decoder,
                const paddle::Tensor &max_dec_len,
                const int max_ngram_size,
                const int max_draft_tokens) {
  auto input_ids_shape = input_ids.shape();
  const int64_t input_ids_stride = input_ids_shape[1];

  const int64_t max_model_len = token_ids_all.shape()[1];

  auto draft_tokens_shape = draft_tokens.shape();
  const int64_t draft_tokens_stride = draft_tokens_shape[1];

  const int64_t max_batch_size = seq_lens_this_time.shape()[0];

  int threshold = 128;
  const char *env_var = getenv("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD");
  if (env_var) {
    threshold = std::stoi(env_var);
  }

  if (input_ids.is_gpu()) {
    auto stream = input_ids.stream();

    // Persistent scratch buffers for Phase 1 → Phase 2 communication.
    // Cached across calls to avoid per-invocation allocation overhead.
    // Write-before-read pattern (Phase 1 writes all elements before
    // Phase 2 reads) means no initialization is needed between calls.
    // Safety: single-threaded Python caller + CUDA stream serialization.
    static paddle::Tensor s_draft_copy;
    static paddle::Tensor s_seqlens_copy;
    static int64_t s_scratch_batch = 0;
    static int64_t s_scratch_stride = 0;

    if (max_batch_size > s_scratch_batch ||
        draft_tokens_stride > s_scratch_stride) {
      s_draft_copy = paddle::empty({max_batch_size, draft_tokens_stride},
                                   paddle::DataType::INT64,
                                   input_ids.place());
      s_seqlens_copy = paddle::empty(
          {max_batch_size}, paddle::DataType::INT32, input_ids.place());
      s_scratch_batch = max_batch_size;
      s_scratch_stride = draft_tokens_stride;
    }
    auto &draft_tokens_copy = s_draft_copy;
    auto &seq_lens_this_time_copy = s_seqlens_copy;

    // Fail-fast: BlockScan Phase 2 requires max_batch_size ≤ block size.
    PD_CHECK(max_batch_size <= NGRAM_GATHER_THREADS,
             "ngram_match: max_batch_size exceeds NGRAM_GATHER_THREADS");

    // Phase 1: parallel search — one block per batch item.
    // Also copies matched tokens to scratch and writes tentative seq_lens.
    ngram_match_search_kernel<<<max_batch_size,
                                NGRAM_BLOCK_THREADS,
                                0,
                                stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        token_ids_all.data<int64_t>(),
        prompt_lens.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        seq_lens_encoder.data<int32_t>(),
        seq_lens_decoder.data<int32_t>(),
        max_dec_len.data<int64_t>(),
        draft_tokens_copy.data<int64_t>(),
        seq_lens_this_time_copy.data<int32_t>(),
        input_ids_stride,
        max_model_len,
        draft_tokens_stride,
        max_batch_size,
        max_ngram_size);

    // Phase 2: BlockScan threshold enforcement + final token copy.
    // <<<1, NGRAM_GATHER_THREADS>>> — all batch items handled by one block.
    ngram_match_gather_kernel<<<1, NGRAM_GATHER_THREADS, 0, stream>>>(
        draft_tokens_copy.data<int64_t>(),
        seq_lens_this_time_copy.data<int32_t>(),
        seq_lens_encoder.data<int32_t>(),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
        draft_tokens_stride,
        max_batch_size,
        threshold);
  } else {
    find_candidate_pred_tokens(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        token_ids_all.data<int64_t>(),
        prompt_lens.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
        const_cast<int32_t *>(seq_lens_encoder.data<int32_t>()),
        const_cast<int32_t *>(seq_lens_decoder.data<int32_t>()),
        const_cast<int64_t *>(max_dec_len.data<int64_t>()),
        input_ids_stride,
        max_model_len,
        draft_tokens_stride,
        max_batch_size,
        max_ngram_size,
        max_draft_tokens);
  }
}

PD_BUILD_STATIC_OP(ngram_match)
    .Inputs({"input_ids",
             "input_ids_len",
             "token_ids_all",
             "prompt_lens",
             "step_idx",
             "draft_token_num",
             "draft_tokens",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "max_dec_len"})
    .Attrs({"max_ngram_size: int", "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
    .SetKernelFn(PD_KERNEL(NgramMatch))
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}});
