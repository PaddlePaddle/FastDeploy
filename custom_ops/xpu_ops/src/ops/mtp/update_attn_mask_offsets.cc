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

std::vector<paddle::Tensor> UpdateAttnMaskOffsets(
    const paddle::Tensor& ids_remove_padding,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& attn_mask_offsets_full,
    const paddle::Tensor& attn_mask_offsets_decoder,
    const paddle::Tensor& is_block_step,
    const paddle::Tensor& decode_states,
    const paddle::Tensor& mask_rollback) {
  int max_model_len = attn_mask_offsets_full.shape()[1];
  int real_bsz = seq_lens_this_time.shape()[0];
  int batch_seq_lens = ids_remove_padding.shape()[0];
  int decode_states_len = decode_states.shape()[1];

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();
  if (ids_remove_padding.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }

  auto attn_mask_offsets = paddle::full({batch_seq_lens * 2},
                                        0,
                                        paddle::DataType::INT32,
                                        ids_remove_padding.place());

  baidu::xpu::api::plugin::update_attn_mask_offsets(
      ctx,
      attn_mask_offsets.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      cu_seqlens_q.data<int>(),
      attn_mask_offsets_full.data<int>(),
      const_cast<int*>(attn_mask_offsets_decoder.data<int>()),
      is_block_step.data<bool>(),
      const_cast<int*>(decode_states.data<int>()),
      const_cast<int*>(mask_rollback.data<int>()),
      real_bsz,
      max_model_len,
      decode_states_len);

  if (ids_remove_padding.is_cpu()) {
    delete ctx;
  }

  return {attn_mask_offsets};
}

PD_BUILD_STATIC_OP(update_attn_mask_offsets)
    .Inputs({"ids_remove_padding",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "cu_seqlens_q",
             "attn_mask_offsets_full",
             "attn_mask_offsets_decoder",
             "is_block_step",
             "decode_states",
             "mask_rollback"})
    .Outputs({"attn_mask_offsets", "decode_states_out", "mask_rollback_out"})
    .SetInplaceMap({{"decode_states", "decode_states_out"},
                    {"mask_rollback", "mask_rollback_out"}})
    .SetKernelFn(PD_KERNEL(UpdateAttnMaskOffsets));
