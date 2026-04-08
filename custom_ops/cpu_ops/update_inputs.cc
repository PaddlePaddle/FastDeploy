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
#include "paddle/extension.h"

void update_inputs_kernel(bool *not_need_stop,
                          int *seq_lens_this_time,
                          int *seq_lens_encoder,
                          int *seq_lens_decoder,
                          int64_t *input_ids,
                          const int64_t *stop_nums,
                          const bool *stop_flags,
                          const bool *is_block_step,
                          const int64_t *next_tokens,
                          const int bsz,
                          const int input_ids_stride) {
  int64_t stop_sum = 0;
  for (int bi = 0; bi < bsz; ++bi) {
    bool stop_flag_now = false;
    int64_t stop_flag_now_int = 0;
    stop_flag_now = stop_flags[bi];
    stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
    auto seq_len_this_time = seq_lens_this_time[bi];
    auto seq_len_encoder = seq_lens_encoder[bi];
    auto seq_len_decoder = seq_lens_decoder[bi];
    seq_lens_decoder[bi] =
        stop_flag_now
            ? 0
            : (seq_len_decoder == 0 ? seq_len_encoder : seq_len_decoder + 1);
    seq_lens_this_time[bi] = stop_flag_now ? 0 : 1;
    seq_lens_encoder[bi] = 0;
    int64_t *input_ids_now = input_ids + bi * input_ids_stride;
    input_ids_now[0] = next_tokens[bi];
    stop_sum += stop_flag_now_int;
  }
  not_need_stop[0] = stop_sum < stop_nums[0];
}

void UpdateInputs(const paddle::Tensor &stop_flags,
                  const paddle::Tensor &not_need_stop,
                  const paddle::Tensor &seq_lens_this_time,
                  const paddle::Tensor &seq_lens_encoder,
                  const paddle::Tensor &seq_lens_decoder,
                  const paddle::Tensor &input_ids,
                  const paddle::Tensor &stop_nums,
                  const paddle::Tensor &next_tokens,
                  const paddle::Tensor &is_block_step) {
  const int bsz = input_ids.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  update_inputs_kernel(const_cast<bool *>(not_need_stop.data<bool>()),
                       const_cast<int *>(seq_lens_this_time.data<int>()),
                       const_cast<int *>(seq_lens_encoder.data<int>()),
                       const_cast<int *>(seq_lens_decoder.data<int>()),
                       const_cast<int64_t *>(input_ids.data<int64_t>()),
                       stop_nums.data<int64_t>(),
                       stop_flags.data<bool>(),
                       is_block_step.data<bool>(),
                       next_tokens.data<int64_t>(),
                       bsz,
                       input_ids_stride);
}

PD_BUILD_STATIC_OP(update_inputs_cpu)
    .Inputs({"stop_flags",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputs));
