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

namespace api = baidu::xpu::api;
void SpecGetStopFlagsMultiSeqs(const paddle::Tensor &accept_tokens,
                               const paddle::Tensor &accept_num,
                               const paddle::Tensor &pre_ids,
                               const paddle::Tensor &step_idx,
                               const paddle::Tensor &stop_flags,
                               const paddle::Tensor &seq_lens,
                               const paddle::Tensor &stop_seqs,
                               const paddle::Tensor &stop_seqs_len,
                               const paddle::Tensor &end_ids) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  if (accept_tokens.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }
  PD_CHECK(accept_tokens.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

  std::vector<int64_t> shape = accept_tokens.shape();
  std::vector<int64_t> stop_seqs_shape = stop_seqs.shape();
  int bs_now = shape[0];
  int stop_seqs_bs = stop_seqs_shape[0];
  int stop_seqs_max_len = stop_seqs_shape[1];
  int pre_ids_len = pre_ids.shape()[1];
  int accept_tokens_len = accept_tokens.shape()[1];

  int r = baidu::xpu::api::plugin::speculate_set_stop_value_multi_seqs(
      ctx,
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      pre_ids.data<int64_t>(),
      step_idx.data<int64_t>(),
      stop_seqs.data<int64_t>(),
      stop_seqs_len.data<int>(),
      seq_lens.data<int>(),
      end_ids.data<int64_t>(),
      bs_now,
      accept_tokens_len,
      stop_seqs_bs,
      stop_seqs_max_len,
      pre_ids_len);
  PD_CHECK(r == 0, "xpu::plugin::speculate_set_stop_value_multi_seqs failed.");
}

PD_BUILD_OP(speculate_set_stop_value_multi_seqs)
    .Inputs({"accept_tokens",
             "accept_num",
             "pre_ids",
             "step_idx",
             "stop_flags",
             "seq_lens",
             "stop_seqs",
             "stop_seqs_len",
             "end_ids"})
    .Outputs({"accept_tokens_out", "stop_flags_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(SpecGetStopFlagsMultiSeqs));
