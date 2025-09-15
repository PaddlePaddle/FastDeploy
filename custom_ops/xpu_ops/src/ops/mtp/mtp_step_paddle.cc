// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace api = baidu::xpu::api;
void MTPStepPaddle(
    const paddle::Tensor &base_model_stop_flags,
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &batch_drop,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &block_tables,  // [bsz, block_num_per_seq]
    const paddle::Tensor &encoder_block_lens,
    const paddle::Tensor &used_list_len,
    const paddle::Tensor &free_list,
    const paddle::Tensor &free_list_len,
    const int block_size,
    const int max_draft_tokens) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();
  if (base_model_stop_flags.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }
  const int bsz = seq_lens_this_time.shape()[0];
  const int block_num_per_seq = block_tables.shape()[1];

  int r = baidu::xpu::api::plugin::mtp_free_and_dispatch_block(
      ctx,
      const_cast<bool *>(base_model_stop_flags.data<bool>()),
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<bool *>(batch_drop.data<bool>()),
      const_cast<int *>(seq_lens_this_time.data<int>()),
      const_cast<int *>(seq_lens_decoder.data<int>()),
      const_cast<int *>(block_tables.data<int>()),
      const_cast<int *>(encoder_block_lens.data<int>()),
      const_cast<int *>(used_list_len.data<int>()),
      const_cast<int *>(free_list.data<int>()),
      const_cast<int *>(free_list_len.data<int>()),
      bsz,
      block_size,
      block_num_per_seq,
      max_draft_tokens);
  PD_CHECK(r == 0, "free_and_dispatch_block failed.");
  if (base_model_stop_flags.is_cpu() && ctx != nullptr) {
    delete ctx;
  }
}

PD_BUILD_STATIC_OP(mtp_step_paddle)
    .Inputs({"base_model_stop_flags",
             "stop_flags",
             "batch_drop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "used_list_len",
             "free_list",
             "free_list_len"})
    .Attrs({"block_size: int", "max_draft_tokens: int"})
    .Outputs({"block_tables_out",
              "stop_flags_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out"})
    .SetInplaceMap({{"block_tables", "block_tables_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"}})
    .SetKernelFn(PD_KERNEL(MTPStepPaddle));
