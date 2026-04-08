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
#include <stdio.h>
#include "paddle/common/flags.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xpu/internal/infra_op.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void SpeculateGetLogits(const paddle::Tensor& draft_logits,
                        const paddle::Tensor& next_token_num,
                        const paddle::Tensor& batch_token_num,
                        const paddle::Tensor& cu_next_token_offset,
                        const paddle::Tensor& cu_batch_token_offset,
                        const paddle::Tensor& logits,
                        const paddle::Tensor& first_token_logits,
                        const paddle::Tensor& seq_lens_this_time,
                        const paddle::Tensor& seq_lens_encoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();
  if (draft_logits.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  const int vocab_size = logits.shape()[1];
  const int real_bsz = seq_lens_this_time.shape()[0];

  baidu::xpu::api::plugin::speculate_get_logits(
      ctx,
      const_cast<float*>(draft_logits.data<float>()),
      const_cast<int*>(next_token_num.data<int>()),
      const_cast<int*>(batch_token_num.data<int>()),
      const_cast<int*>(cu_next_token_offset.data<int>()),
      const_cast<int*>(cu_batch_token_offset.data<int>()),
      logits.data<float>(),
      first_token_logits.data<float>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      real_bsz,
      vocab_size);
  if (draft_logits.is_cpu()) {
    delete ctx;
  }
}

PD_BUILD_STATIC_OP(speculate_get_logits)
    .Inputs({"draft_logits",
             "next_token_num",
             "batch_token_num",
             "cu_next_token_offset",
             "cu_batch_token_offset",
             "logits",
             "first_token_logits",
             "seq_lens_this_time",
             "seq_lens_encoder"})
    .Outputs({"draft_logits_out",
              "batch_token_num_out",
              "cu_batch_token_offset_out"})
    .SetInplaceMap({{"draft_logits", "draft_logits_out"},
                    {"batch_token_num", "batch_token_num_out"},
                    {"cu_batch_token_offset", "cu_batch_token_offset_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateGetLogits));
