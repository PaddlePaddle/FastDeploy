// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"

/**
 * @file naive_update_model_status.cu
 * @brief Post-sampling update for NAIVE speculative decoding mode.
 *
 * Responsibilities (one thread per batch slot, <<<1, 1024>>>):
 *   1. Scatter sampled token into accept_tokens[i, 0] using cu_seqlens_q_output
 *      to map from the packed next_tokens array to the per-batch output.
 *   2. Set accept_num[i] = 1 for running slots (seq_lens_this_time > 0), else
 * 0.
 *   3. Set seq_lens_this_time[i] = 1 for running, 0 for stopped/paused.
 *
 * Running slots are identified by seq_lens_this_time[i] > 0, which is already
 * zeroed for stopped/paused slots by pre_process before this kernel runs.
 *
 * The packed next_tokens layout mirrors cu_seqlens_q_output:
 *   next_tokens[cu_seqlens_q_output[i] .. cu_seqlens_q_output[i+1]-1]
 *   are the output tokens for request i (exactly 1 per running slot in naive
 *   decode; 0 for stopped/encoder-only slots).
 */
template <int THREADBLOCK_SIZE>
__global__ void naive_update_model_status_kernel(int64_t *accept_tokens,
                                                 int *accept_num,
                                                 int *seq_lens_this_time,
                                                 const int64_t *next_tokens,
                                                 const int *cu_seqlens_q_output,
                                                 int real_bsz,
                                                 int max_step_tokens) {
  int bid = threadIdx.x;

  if (bid >= real_bsz) return;
  if (seq_lens_this_time[bid] > 0) {
    // Write the last (and only) sampled token to accept_tokens[bid, 0]
    accept_tokens[bid * max_step_tokens] =
        next_tokens[cu_seqlens_q_output[bid + 1] - 1];
    accept_num[bid] = 1;
    seq_lens_this_time[bid] = 1;
  } else {
    accept_num[bid] = 0;
    seq_lens_this_time[bid] = 0;
  }
}

void NaiveUpdateModelStatus(const paddle::Tensor &accept_tokens,
                            const paddle::Tensor &accept_num,
                            const paddle::Tensor &seq_lens_this_time,
                            const paddle::Tensor &next_tokens,
                            const paddle::Tensor &cu_seqlens_q_output) {
  constexpr int kBlockSize = 1024;
  const int real_bsz = seq_lens_this_time.shape()[0];
  PADDLE_ENFORCE_LE(
      real_bsz,
      kBlockSize,
      phi::errors::InvalidArgument(
          "naive_update_model_status: real_bsz (%d) must be <= %d.",
          real_bsz,
          kBlockSize));
  const int max_step_tokens = accept_tokens.shape()[1];
  auto cu_stream = seq_lens_this_time.stream();

  naive_update_model_status_kernel<kBlockSize><<<1, kBlockSize, 0, cu_stream>>>(
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      const_cast<int *>(seq_lens_this_time.data<int>()),
      next_tokens.data<int64_t>(),
      cu_seqlens_q_output.data<int>(),
      real_bsz,
      max_step_tokens);
}

PD_BUILD_STATIC_OP(naive_update_model_status)
    .Inputs({"accept_tokens",
             "accept_num",
             "seq_lens_this_time",
             "next_tokens",
             "cu_seqlens_q_output"})
    .Attrs({})
    .Outputs({"accept_tokens_out", "accept_num_out", "seq_lens_this_time_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}})
    .SetKernelFn(PD_KERNEL(NaiveUpdateModelStatus));
