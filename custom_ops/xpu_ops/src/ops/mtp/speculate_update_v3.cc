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
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

void SpeculateUpdateV3(const paddle::Tensor &seq_lens_encoder,
                       const paddle::Tensor &seq_lens_decoder,
                       const paddle::Tensor &not_need_stop,
                       const paddle::Tensor &draft_tokens,
                       const paddle::Tensor &actual_draft_token_nums,
                       const paddle::Tensor &accept_tokens,
                       const paddle::Tensor &accept_num,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens_this_time,
                       const paddle::Tensor &is_block_step,
                       const paddle::Tensor &stop_nums) {
  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  auto max_draft_tokens = draft_tokens.shape()[1];

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  if (draft_tokens.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  auto not_need_stop_xpu = not_need_stop.copy_to(stop_flags.place(), false);
  int r = baidu::xpu::api::plugin::speculate_update_v3(
      ctx,
      const_cast<int *>(seq_lens_encoder.data<int>()),
      const_cast<int *>(seq_lens_decoder.data<int>()),
      const_cast<bool *>(not_need_stop_xpu.data<bool>()),
      const_cast<int64_t *>(draft_tokens.data<int64_t>()),
      const_cast<int *>(actual_draft_token_nums.data<int>()),
      accept_tokens.data<int64_t>(),
      accept_num.data<int>(),
      stop_flags.data<bool>(),
      seq_lens_this_time.data<int>(),
      is_block_step.data<bool>(),
      stop_nums.data<int64_t>(),
      real_bsz,
      max_bsz,
      max_draft_tokens);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "speculate_update_v3");

  auto not_need_stop_cpu =
      not_need_stop_xpu.copy_to(not_need_stop.place(), true);
  bool *not_need_stop_data = const_cast<bool *>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_STATIC_OP(speculate_update_v3)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "not_need_stop",
             "draft_tokens",
             "actual_draft_token_nums",
             "accept_tokens",
             "accept_num",
             "stop_flags",
             "seq_lens_this_time",
             "is_block_step",
             "stop_nums"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "not_need_stop_out",
              "draft_tokens_out",
              "actual_draft_token_nums_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"draft_tokens", "draft_tokens_out"},
                    {"actual_draft_token_nums", "actual_draft_token_nums_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateUpdateV3));
