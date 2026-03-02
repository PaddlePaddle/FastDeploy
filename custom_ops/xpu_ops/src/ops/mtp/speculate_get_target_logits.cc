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

void SpeculateGetTargetLogits(const paddle::Tensor& target_logits,
                              const paddle::Tensor& logits,
                              const paddle::Tensor& cu_batch_token_offset,
                              const paddle::Tensor& ori_cu_batch_token_offset,
                              const paddle::Tensor& seq_lens_this_time,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& accept_num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();

  if (seq_lens_this_time.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }

  const int vocab_size = logits.shape()[1];
  const int real_bsz = seq_lens_this_time.shape()[0];
  int r = baidu::xpu::api::plugin::speculate_get_target_logits(
      ctx,
      const_cast<float*>(target_logits.data<float>()),
      logits.data<float>(),
      cu_batch_token_offset.data<int>(),
      ori_cu_batch_token_offset.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      accept_num.data<int>(),
      vocab_size,
      real_bsz);
  PD_CHECK(r == 0, "speculate_get_target_logits  failed.");
}

PD_BUILD_STATIC_OP(speculate_get_target_logits)
    .Inputs({"target_logits",
             "logits",
             "cu_batch_token_offset",
             "ori_cu_batch_token_offset",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "accept_num"})
    .Outputs({"target_logits_out"})
    .SetInplaceMap({{"target_logits", "target_logits_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateGetTargetLogits));
