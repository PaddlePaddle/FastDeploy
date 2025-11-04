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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void LimitThinkingContentLengthV2(const paddle::Tensor& next_tokens,
                                  const paddle::Tensor& max_think_lens,
                                  const paddle::Tensor& step_idx,
                                  const paddle::Tensor& limit_think_status,
                                  const paddle::Tensor& stop_flags,
                                  const int64_t think_end_id,
                                  const int64_t line_break_id) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int batch_size = next_tokens.shape()[0];
  int r = baidu::xpu::api::plugin::limit_thinking_content_length_kernel_v2(
      xpu_ctx->x_context(),
      const_cast<int64_t*>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      step_idx.data<int64_t>(),
      const_cast<int*>(limit_think_status.data<int>()),
      stop_flags.data<bool>(),
      think_end_id,
      line_break_id,
      batch_size);
  PD_CHECK(r == 0,
           "baidu::xpu::api::plugin::limit_thinking_content_length_kernel_v2 "
           "failed.");
}

PD_BUILD_STATIC_OP(limit_thinking_content_length_v2)
    .Inputs({"next_tokens",
             "max_think_lens",
             "step_idx",
             "limit_think_status",
             "stop_flags"})
    .Attrs({"think_end_id: int64_t", "line_break_id: int64_t"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(LimitThinkingContentLengthV2));
