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

void SpeculateSetValueByFlagsAndIdx(const paddle::Tensor &pre_ids_all,
                                    const paddle::Tensor &accept_tokens,
                                    const paddle::Tensor &accept_num,
                                    const paddle::Tensor &stop_flags,
                                    const paddle::Tensor &seq_lens_this_time,
                                    const paddle::Tensor &seq_lens_encoder,
                                    const paddle::Tensor &seq_lens_decoder,
                                    const paddle::Tensor &step_idx) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();

  //   auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  if (pre_ids_all.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }
  std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
  int bs = seq_lens_this_time.shape()[0];
  int length = pre_ids_all_shape[1];
  int max_draft_tokens = accept_tokens.shape()[1];

  int r = baidu::xpu::api::plugin::speculate_set_value_by_flag_and_id(
      ctx,
      const_cast<int64_t *>(pre_ids_all.data<int64_t>()),
      accept_tokens.data<int64_t>(),
      accept_num.data<int>(),
      stop_flags.data<bool>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      step_idx.data<int64_t>(),
      bs,
      length,
      max_draft_tokens);
  PD_CHECK(r == 0, "speculate_clear_accept_nums_kernel  failed.");
}

PD_BUILD_STATIC_OP(speculate_set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all",
             "accept_tokens",
             "accept_num",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSetValueByFlagsAndIdx));
