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

void SpeculateInsertFirstToken(const paddle::Tensor& token_ids,
                               const paddle::Tensor& accept_tokens,
                               const paddle::Tensor& next_tokens,
                               const paddle::Tensor& cu_next_token_offset,
                               const paddle::Tensor& cu_batch_token_offset,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_encoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();

  if (seq_lens_this_time.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }

  const int max_draft_tokens = accept_tokens.shape()[1];
  const int real_bsz = seq_lens_this_time.shape()[0];
  int r = fastdeploy::plugin::speculate_insert_first_token(
      ctx,
      const_cast<int64_t*>(token_ids.data<int64_t>()),
      accept_tokens.data<int64_t>(),
      next_tokens.data<int64_t>(),
      cu_next_token_offset.data<int>(),
      cu_batch_token_offset.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      max_draft_tokens,
      real_bsz);
  PD_CHECK(r == 0, "speculate_insert_first_token failed.");
}

PD_BUILD_STATIC_OP(speculate_insert_first_token)
    .Inputs({"token_ids",
             "accept_tokens",
             "next_tokens",
             "cu_next_token_offset",
             "cu_batch_token_offset",
             "seq_lens_this_time",
             "seq_lens_encoder"})
    .Outputs({"token_ids_out"})
    .SetInplaceMap({{"token_ids", "token_ids_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateInsertFirstToken));
