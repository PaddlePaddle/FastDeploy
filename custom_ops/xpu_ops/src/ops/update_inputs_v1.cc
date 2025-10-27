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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

void UpdateInputsV1(const paddle::Tensor& stop_flags,
                    const paddle::Tensor& not_need_stop,  // only on cpu
                    const paddle::Tensor& seq_lens_this_time,
                    const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& step_seq_lens_decoder,
                    const paddle::Tensor& prompt_lens,
                    const paddle::Tensor& topk_ids,
                    const paddle::Tensor& input_ids,
                    const paddle::Tensor& block_tables,
                    const paddle::Tensor& stop_nums,
                    const paddle::Tensor& next_tokens,
                    const paddle::Tensor& is_block_step,
                    const int block_size) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int max_bsz = stop_flags.shape()[0];
  const int now_bsz = seq_lens_this_time.shape()[0];
  // std::cout << "now_bsz: " << now_bsz << std::endl;
  const int input_ids_stride = input_ids.shape()[1];
  const int block_num_per_seq = block_tables.shape()[1];
  auto not_need_stop_gpu = not_need_stop.copy_to(stop_flags.place(), false);
  int r = baidu::xpu::api::plugin::update_inputs_v1(
      xpu_ctx->x_context(),
      const_cast<bool*>(not_need_stop_gpu.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int*>(step_seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(prompt_lens.data<int64_t>()),
      const_cast<int64_t*>(topk_ids.data<int64_t>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      const_cast<int*>(block_tables.data<int>()),
      stop_nums.data<int64_t>(),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<bool*>(is_block_step.data<bool>()),
      next_tokens.data<int64_t>(),
      now_bsz,
      max_bsz,
      input_ids_stride,
      block_num_per_seq,
      block_size);
  PD_CHECK(r == 0, "baidu::xpu::api::plugin::update_inputs_kernel_v1 failed.");
  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), false);
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_OP(update_inputs_v1)
    .Inputs({"stop_flags",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_seq_lens_decoder",
             "prompt_lens",
             "topk_ids",
             "input_ids",
             "block_tables",
             "stop_nums",
             "next_tokens",
             "is_block_step"})
    .Attrs({"block_size: int"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_seq_lens_decoder_out",
              "topk_ids_out",
              "input_ids_out",
              "stop_flags_out",
              "is_block_step_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"topk_ids", "topk_ids_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"step_seq_lens_decoder", "step_seq_lens_decoder_out"},
                    {"is_block_step", "is_block_step_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputsV1));
