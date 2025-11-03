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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void SpeculateClearAcceptNums(const paddle::Tensor& accept_num,
                              const paddle::Tensor& seq_lens_decoder) {
  // printf("enter clear \n");
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  const int max_bsz = seq_lens_decoder.shape()[0];
  int r = baidu::xpu::api::plugin::speculate_clear_accept_nums(
      xpu_ctx->x_context(),
      const_cast<int*>(accept_num.data<int>()),
      seq_lens_decoder.data<int>(),
      max_bsz);
  PD_CHECK(r == 0, "speculate_clear_accept_nums_kernel  failed.");
}

PD_BUILD_STATIC_OP(speculate_clear_accept_nums)
    .Inputs({"accept_num", "seq_lens_decoder"})
    .Outputs({"seq_lens_decoder_out"})
    .SetInplaceMap({{"seq_lens_decoder", "seq_lens_decoder_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateClearAcceptNums));
