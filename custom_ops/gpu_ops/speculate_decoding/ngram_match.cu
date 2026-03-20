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

#include <cstdlib>
#include <string>
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// GPU kernel for ngram matching — eliminates CPU↔GPU data copies.
// Uses single-thread execution to preserve sequential threshold semantics
// across batch items. The performance win comes from zero-copy data access:
// all tensors stay on GPU, removing the forced CUDA stream synchronization
// that the CPU path requires.
__global__ void ngram_match_kernel(const int64_t *input_ids,
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
                                   int max_ngram_size,
                                   int max_draft_tokens_param,
                                   int threshold) {
  // Phase 1: Count active batch items for threshold calculation
  int unprocessed_batch_size = 0;
  for (int batch_idx = 0; batch_idx < max_batch_size; batch_idx++) {
    if (seq_lens_encoder[batch_idx] > 0 || seq_lens_decoder[batch_idx] > 0) {
      unprocessed_batch_size++;
    }
  }

  // Phase 2: Process each batch item sequentially (threshold creates
  // inter-batch data dependency via running sum of seq_lens_this_time)
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

    const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
    int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
    const int64_t *cur_pre_ids =
        token_ids_all + batch_idx * max_model_len + prompt_lens[batch_idx];
    const int64_t cur_step_idx = step_idx[batch_idx];
    const int64_t cur_input_ids_len = input_ids_len[batch_idx];
    seq_lens_this_time[batch_idx] = 1;
    unprocessed_batch_size--;

    // Running sum includes current batch_idx (just set to 1 above)
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

    for (int ngram_size = max_ngram_size; ngram_size > 0; --ngram_size) {
      if (cur_step_idx < ngram_size) continue;

      const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

      // Search in input_ids (prompt tokens)
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
              min(start_idx + static_cast<int64_t>(max_draft_tokens),
                  cur_input_ids_len);
          if (start_idx >= end_idx) continue;

          int64_t n = end_idx - start_idx;
          seq_lens_this_time[batch_idx] = static_cast<int32_t>(n + 1);
          for (int64_t k = 0; k < n; k++) {
            cur_draft_tokens[1 + k] = cur_input_ids[start_idx + k];
          }
          ngram_size = 0;
          match_input = true;
          break;
        }
      }

      // Search in pre_ids (previously generated tokens)
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
                min(start_idx + static_cast<int64_t>(max_draft_tokens),
                    cur_step_idx);
            int64_t n = end_idx - start_idx;
            if (start_idx >= end_idx) continue;

            seq_lens_this_time[batch_idx] = static_cast<int32_t>(n + 1);
            for (int64_t k = 0; k < n; k++) {
              cur_draft_tokens[1 + k] = cur_pre_ids[start_idx + k];
            }
            ngram_size = 0;
            break;
          }
        }
      }
    }
  }
}

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

  ngram_match_kernel<<<1, 1, 0, input_ids.stream()>>>(
      input_ids.data<int64_t>(),
      input_ids_len.data<int64_t>(),
      token_ids_all.data<int64_t>(),
      prompt_lens.data<int64_t>(),
      step_idx.data<int64_t>(),
      draft_token_num.data<int>(),
      const_cast<int64_t *>(draft_tokens.data<int64_t>()),
      const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
      seq_lens_encoder.data<int32_t>(),
      seq_lens_decoder.data<int32_t>(),
      max_dec_len.data<int64_t>(),
      input_ids_stride,
      max_model_len,
      draft_tokens_stride,
      max_batch_size,
      max_ngram_size,
      max_draft_tokens,
      threshold);
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
