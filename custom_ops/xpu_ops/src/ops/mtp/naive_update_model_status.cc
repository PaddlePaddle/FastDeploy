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
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

void NaiveUpdateModelStatus(const paddle::Tensor &accept_tokens,
                            const paddle::Tensor &accept_num,
                            const paddle::Tensor &seq_lens_this_time,
                            const paddle::Tensor &next_tokens,
                            const paddle::Tensor &cu_seqlens_q_output) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();

  std::unique_ptr<baidu::xpu::api::Context> cpu_ctx;
  if (accept_tokens.place().GetType() == phi::AllocationType::CPU) {
    cpu_ctx = std::make_unique<baidu::xpu::api::Context>(baidu::xpu::api::kCPU);
    ctx = cpu_ctx.get();
  }

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

  int r = fastdeploy::plugin::naive_update_model_status(
      ctx,
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      const_cast<int *>(seq_lens_this_time.data<int>()),
      next_tokens.data<int64_t>(),
      cu_seqlens_q_output.data<int>(),
      real_bsz,
      max_step_tokens);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "naive_update_model_status");
}

PD_BUILD_STATIC_OP(naive_update_model_status)
    .Inputs({"accept_tokens",
             "accept_num",
             "seq_lens_this_time",
             "next_tokens",
             "cu_seqlens_q_output"})
    .Outputs({"accept_tokens_out", "accept_num_out", "seq_lens_this_time_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}})
    .SetKernelFn(PD_KERNEL(NaiveUpdateModelStatus));
