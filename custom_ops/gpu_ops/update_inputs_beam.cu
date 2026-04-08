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

#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void update_inputs_beam_kernel(int* seq_lens_this_time,
                                          int* seq_lens_encoder,
                                          int64_t* input_ids,
                                          float* logits,
                                          const int bsz,
                                          const int seq_len,
                                          const int hidden_size,
                                          const int beam_width) {
  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;

  if (thread_idx < bsz) {
    int bsz_index = thread_idx / beam_width * beam_width;
    if (seq_lens_encoder[bsz_index] > 0) {
      if (block_idx == 0) {
        seq_lens_this_time[thread_idx] = seq_lens_this_time[bsz_index];
        seq_lens_encoder[thread_idx] = seq_lens_encoder[bsz_index];
      }
      if (block_idx < seq_len) {
        input_ids[thread_idx * seq_len + block_idx] =
            input_ids[bsz_index * seq_len + block_idx];
      }

      logits[thread_idx * hidden_size + block_idx] =
          logits[bsz_index * hidden_size + block_idx];
    }
  }
  __syncthreads();
}

void UpdateInputsBeam(const paddle::Tensor& beam_width,
                      const paddle::Tensor& seq_lens_this_time,
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& input_ids,
                      const paddle::Tensor& logits) {
  int beam_width_scalar = beam_width.data<int>()[0];

  if (beam_width_scalar > 1) {
    const int bsz = seq_lens_this_time.shape()[0];
    const int seq_len = input_ids.shape()[1];
    const int hidden_size = logits.shape()[1];

    update_inputs_beam_kernel<1024>
        <<<hidden_size, 1024, 0, input_ids.stream()>>>(
            const_cast<int*>(seq_lens_this_time.data<int>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int64_t*>(input_ids.data<int64_t>()),
            const_cast<float*>(logits.data<float>()),
            bsz,
            seq_len,
            hidden_size,
            beam_width_scalar);
  }
}

PD_BUILD_STATIC_OP(update_inputs_beam)
    .Inputs({"beam_width",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "input_ids",
             "logits"})
    .Outputs({"seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "input_ids_out",
              "logits_out"})
    .SetInplaceMap({{"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"input_ids", "input_ids_out"},
                    {"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputsBeam));
