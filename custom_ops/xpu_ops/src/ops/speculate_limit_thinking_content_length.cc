// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

void SpeculateLimitThinkingContentLength(const paddle::Tensor& next_tokens,
                                         const paddle::Tensor& max_think_lens,
                                         const paddle::Tensor& max_reply_lens,
                                         const paddle::Tensor& step_idx,
                                         const paddle::Tensor& limit_status,
                                         const paddle::Tensor& accept_num,
                                         const paddle::Tensor& stop_flags,
                                         const paddle::Tensor& eos_token_ids,
                                         const paddle::Tensor& inject_token_ids,
                                         const int64_t think_end_id,
                                         const bool splitwise_role_is_decode) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int batch_size = next_tokens.shape()[0];
  const int tokens_per_step = next_tokens.shape()[1];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  const int inject_len = inject_token_ids.shape()[0];

  int r =
      baidu::xpu::api::plugin::speculate_limit_thinking_content_length_kernel(
          xpu_ctx->x_context(),
          const_cast<int64_t*>(next_tokens.data<int64_t>()),
          max_think_lens.data<int>(),
          const_cast<int*>(max_reply_lens.data<int>()),
          const_cast<int64_t*>(step_idx.data<int64_t>()),
          eos_token_ids.data<int64_t>(),
          const_cast<int*>(limit_status.data<int>()),
          const_cast<int*>(accept_num.data<int>()),
          stop_flags.data<bool>(),
          think_end_id,
          (inject_len > 0) ? inject_token_ids.data<int64_t>() : nullptr,
          tokens_per_step,
          batch_size,
          eos_token_id_len,
          inject_len,
          splitwise_role_is_decode);
  PD_CHECK(r == 0,
           "baidu::xpu::api::plugin::"
           "speculate_limit_thinking_content_length_kernel failed.");
}

PD_BUILD_STATIC_OP(speculate_limit_thinking_content_length)
    .Inputs({"next_tokens",
             "max_think_lens",
             "max_reply_lens",
             "step_idx",
             "limit_status",
             "accept_num",
             "stop_flags",
             "eos_token_ids",
             "inject_token_ids"})
    .Attrs({"think_end_id: int64_t", "splitwise_role_is_decode: bool"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateLimitThinkingContentLength));
