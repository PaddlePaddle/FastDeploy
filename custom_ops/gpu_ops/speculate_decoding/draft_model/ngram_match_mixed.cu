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
#include "../ngram_match_common.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// ============================================================
// Phase 1 mixed search kernel — one block per batch item.
// Also copies tentative matched tokens to scratch buffers.
// ============================================================
__global__ void ngram_match_mixed_search_kernel(
    const int64_t *input_ids,
    const int64_t *input_ids_len,
    const int64_t *pre_ids,
    const int64_t *step_idx,
    const int *draft_token_num,
    const int32_t *seq_lens_this_time,
    const int64_t *max_dec_len,
    int64_t *draft_tokens_copy,
    int32_t *seq_lens_this_time_copy,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t draft_tokens_stride,
    int64_t max_batch_size,
    int max_ngram_size,
    int min_ngram_size,
    int max_draft_tokens_param) {
  int batch_idx = blockIdx.x;
  if (batch_idx >= max_batch_size) return;

  __shared__ int64_t s_min_pos;

  const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];

  if (threadIdx.x == 0) {
    // Default: keep the original seq_lens_this_time (no ngram match)
    seq_lens_this_time_copy[batch_idx] = ori_seq_len_this_time;
  }
  __syncthreads();

  // Skip batch items with no active tokens
  if (ori_seq_len_this_time == 0) return;

  // Compute max_draft_tokens for this batch item.
  // Split into explicit steps to avoid negative intermediate values.
  int64_t draft_budget =
      static_cast<int64_t>(max_draft_tokens_param) - ori_seq_len_this_time + 1;
  int64_t remaining_dec = max_dec_len[batch_idx] - step_idx[batch_idx] - 1;
  if (draft_budget <= 0 || remaining_dec <= 0) return;
  int max_draft_tokens = static_cast<int>(min(draft_budget, remaining_dec));

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
      int64_t start_idx = pos + ngram_size;
      int64_t end_idx = min(start_idx + static_cast<int64_t>(max_draft_tokens),
                            cur_input_ids_len);
      if (threadIdx.x == 0 && start_idx < end_idx) {
        // Tentative token copy to scratch
        int64_t n = end_idx - start_idx;
        seq_lens_this_time_copy[batch_idx] =
            static_cast<int32_t>(ori_seq_len_this_time + n);
        int64_t *dst = draft_tokens_copy + batch_idx * draft_tokens_stride;
        for (int64_t k = 0; k < n; k++) {
          dst[ori_seq_len_this_time + k] = cur_input_ids[start_idx + k];
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
        seq_lens_this_time_copy[batch_idx] =
            static_cast<int32_t>(ori_seq_len_this_time + n);
        int64_t *dst = draft_tokens_copy + batch_idx * draft_tokens_stride;
        for (int64_t k = 0; k < n; k++) {
          dst[ori_seq_len_this_time + k] = cur_pre_ids[start_idx + k];
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
// Phase 2 mixed gather kernel — BlockScan threshold + copy
//   <<<1, NGRAM_GATHER_THREADS>>>
//
// Reads tentative allocations from Phase 1 scratch buffers,
// computes prefix sums to enforce the global threshold, then
// writes final seq_lens_this_time and copies draft tokens.
// The mixed variant respects ori_seq_len_this_time (MTP tokens).
// ============================================================
__global__ void ngram_match_mixed_gather_kernel(
    const int64_t *draft_tokens_copy,
    const int32_t *seq_lens_this_time_copy,
    const int32_t *seq_lens_this_time_orig,
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

  // Load tentative total token count from Phase 1
  int tentative = 0;
  int is_active = 0;
  if (tid < max_batch_size) {
    tentative = seq_lens_this_time_copy[tid];
    is_active = (tentative > 0) ? 1 : 0;
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
    if (tentative == 0) {
      seq_lens_this_time[tid] = 0;
      return;
    }

    int ori = seq_lens_this_time_orig[tid];
    int ngram_tokens = tentative - ori;  // tokens added by ngram match

    int exclusive_token_prefix = token_prefix - tentative;
    int remaining_active = s_total_active - active_prefix;

    // Budget: threshold minus tokens already allocated before me,
    // minus at-least-1 reservation for every active item after me.
    int budget = threshold - exclusive_token_prefix - remaining_active;

    int actual;
    if (budget <= ori) {
      // Can't even keep all MTP base tokens — keep original only
      actual = ori;
    } else {
      int ngram_budget = budget - ori;
      actual = ori + min(ngram_tokens, ngram_budget);
    }
    actual = min(actual, tentative);

    seq_lens_this_time[tid] = actual;

    // Copy ngram draft tokens from scratch to output
    int ngram_to_copy = actual - ori;
    if (ngram_to_copy > 0) {
      int64_t *dst = draft_tokens + tid * draft_tokens_stride;
      const int64_t *src = draft_tokens_copy + tid * draft_tokens_stride;
      for (int k = 0; k < ngram_to_copy; k++) {
        dst[ori + k] = src[ori + k];
      }
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
    // Split into explicit int64_t steps to avoid negative intermediate values.
    int64_t draft_budget =
        static_cast<int64_t>(max_draft_tokens) - ori_seq_len_this_time + 1;
    int64_t remaining_dec = max_dec_len[batch_idx] - step_idx[batch_idx] - 1;

    if (ori_seq_len_this_time == 0 || draft_budget <= 0 || remaining_dec <= 0) {
      continue;
    }
    int max_draft_tokens_query =
        static_cast<int>(std::min(draft_budget, remaining_dec));

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
//          search within each batch item (NGRAM_BLOCK_THREADS threads
//          per block).  Also copies matched draft tokens to scratch.
// Phase 2: <<<1, NGRAM_GATHER_THREADS>>> — CUB BlockScan prefix-sum
//          threshold enforcement + final token copy.
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

    // NOTE: GPU path does not pass seq_lens_decoder to kernels — the mixed
    // variant uses ori_seq_len_this_time == 0 to skip inactive items. This
    // matches CPU behavior under the invariant that seq_lens_decoder > 0 iff
    // ori_seq_len_this_time > 0 (holds during normal MTP decoding). The CPU
    // path counts seq_lens_decoder > 0 for threshold budget; the GPU scan
    // counts tentative > 0, which is equivalent under this invariant.

    // Allocate scratch buffers for Phase 1 → Phase 2 communication

    // Scratch copy of draft_tokens (Phase 1 writes tentative tokens here)
    auto draft_tokens_copy =
        paddle::empty({max_batch_size, draft_tokens_stride},
                      paddle::DataType::INT64,
                      input_ids.place());

    // Scratch copy of seq_lens_this_time (Phase 1 writes tentative counts)
    auto seq_lens_this_time_copy = paddle::empty(
        {max_batch_size}, paddle::DataType::INT32, input_ids.place());

    // Save a copy of original seq_lens_this_time for Phase 2
    // (Phase 1 reads from the original, Phase 2 needs ori values)
    auto seq_lens_this_time_orig = paddle::empty(
        {max_batch_size}, paddle::DataType::INT32, input_ids.place());
    cudaMemcpyAsync(seq_lens_this_time_orig.data<int32_t>(),
                    seq_lens_this_time.data<int32_t>(),
                    max_batch_size * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice,
                    stream);

    // Fail-fast: BlockScan Phase 2 requires max_batch_size ≤ block size.
    PD_CHECK(max_batch_size <= NGRAM_GATHER_THREADS,
             "hybrid_mtp_ngram: max_batch_size exceeds NGRAM_GATHER_THREADS");

    // Phase 1: parallel search — one block per batch item.
    // Also copies matched tokens to scratch and writes tentative seq_lens.
    ngram_match_mixed_search_kernel<<<max_batch_size,
                                      NGRAM_BLOCK_THREADS,
                                      0,
                                      stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        seq_lens_this_time.data<int32_t>(),
        max_dec_len.data<int64_t>(),
        draft_tokens_copy.data<int64_t>(),
        seq_lens_this_time_copy.data<int32_t>(),
        input_ids_stride,
        pre_ids_stride,
        draft_tokens_stride,
        max_batch_size,
        max_ngram_size,
        min_ngram_size,
        max_draft_tokens);

    // Phase 2: BlockScan threshold enforcement + final token copy.
    // <<<1, NGRAM_GATHER_THREADS>>> — all batch items handled by one block.
    ngram_match_mixed_gather_kernel<<<1, NGRAM_GATHER_THREADS, 0, stream>>>(
        draft_tokens_copy.data<int64_t>(),
        seq_lens_this_time_copy.data<int32_t>(),
        seq_lens_this_time_orig.data<int32_t>(),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
        draft_tokens_stride,
        max_batch_size,
        threshold);
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
