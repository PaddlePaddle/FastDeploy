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
#include <xft/xdnn_plugin.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void DraftModelPostprocess(const paddle::Tensor& base_model_draft_tokens,
                           const paddle::Tensor& base_model_seq_lens_this_time,
                           const paddle::Tensor& base_model_seq_lens_encoder,
                           const paddle::Tensor& base_model_stop_flags) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  int real_bsz = base_model_draft_tokens.shape()[0];
  int base_model_draft_token_len = base_model_draft_tokens.shape()[1];
  int r = baidu::xpu::api::plugin::draft_model_postprocess(
      xpu_ctx->x_context(),
      const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
      const_cast<int*>(base_model_seq_lens_this_time.data<int>()),
      const_cast<int*>(base_model_seq_lens_encoder.data<int>()),
      const_cast<bool*>(base_model_stop_flags.data<bool>()),
      real_bsz,
      base_model_draft_token_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "");
}

PD_BUILD_STATIC_OP(draft_model_postprocess)
    .Inputs({"base_model_draft_tokens",
             "base_model_seq_lens_this_time",
             "base_model_seq_lens_encoder",
             "base_model_stop_flags"})
    .Outputs({"base_model_draft_tokens_out",
              "base_model_seq_lens_this_time_out",
              "base_model_stop_flags_out"})
    .SetInplaceMap({{"base_model_draft_tokens", "base_model_draft_tokens_out"},
                    {"base_model_seq_lens_this_time",
                     "base_model_seq_lens_this_time_out"},
                    {"base_model_stop_flags", "base_model_stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPostprocess));
