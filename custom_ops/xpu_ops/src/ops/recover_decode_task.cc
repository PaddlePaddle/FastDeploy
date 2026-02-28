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

void RecoverDecodeTask(
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &step_seq_lens_decoder,
    const paddle::Tensor &block_tables,
    const paddle::Tensor &is_block_step,
    const paddle::optional<paddle::Tensor> &draft_tokens,
    const paddle::optional<paddle::Tensor> &step_draft_tokens,
    const paddle::optional<paddle::Tensor> &step_seq_lens_this_time,
    const int block_size,
    const int max_draft_tokens) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  const int bsz = seq_lens_this_time.shape()[0];
  const int block_num_per_seq = block_tables.shape()[1];
  int r = 0;
  if (draft_tokens) {
    const int draft_tokens_len = draft_tokens.get_ptr()->shape()[1];
    r = baidu::xpu::api::plugin::recover_spec_decode_task(
        xpu_ctx->x_context(),
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<int *>(seq_lens_this_time.data<int>()),
        const_cast<int *>(seq_lens_encoder.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<int *>(step_seq_lens_decoder.data<int>()),
        const_cast<int *>(block_tables.data<int>()),
        const_cast<bool *>(is_block_step.data<bool>()),
        const_cast<int64_t *>(draft_tokens.get_ptr()->data<int64_t>()),
        step_draft_tokens.get_ptr()->data<int64_t>(),
        step_seq_lens_this_time.get_ptr()->data<int>(),
        bsz,
        block_num_per_seq,
        block_size,
        draft_tokens_len,
        max_draft_tokens * 2 + 1);
  } else {
    r = baidu::xpu::api::plugin::recover_decode_task(
        xpu_ctx->x_context(),
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<int *>(seq_lens_this_time.data<int>()),
        const_cast<int *>(seq_lens_encoder.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<int *>(step_seq_lens_decoder.data<int>()),
        const_cast<int *>(block_tables.data<int>()),
        const_cast<bool *>(is_block_step.data<bool>()),
        bsz,
        block_num_per_seq,
        block_size);
  }
  PD_CHECK(r == 0, "baidu::xpu::api::plugin::recover_decode_task failed.");
}

PD_BUILD_STATIC_OP(recover_decode_task)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_seq_lens_decoder",
             "block_tables",
             "is_block_step",
             paddle::Optional("draft_tokens"),
             paddle::Optional("step_draft_tokens"),
             paddle::Optional("step_seq_lens_this_time")})
    .Attrs({"block_size: int", "max_draft_tokens: int"})
    .Outputs({"seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "stop_flags_out",
              "is_block_step_out"})
    .SetInplaceMap({{"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"is_block_step", "is_block_step_out"}})
    .SetKernelFn(PD_KERNEL(RecoverDecodeTask));
