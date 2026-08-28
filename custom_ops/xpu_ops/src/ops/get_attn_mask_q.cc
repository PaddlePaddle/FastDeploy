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

std::vector<paddle::Tensor> GetAttnMaskQ(
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const paddle::optional<paddle::Tensor>& attn_mask_kv,
    const int kv_token_num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  baidu::xpu::api::Context* ctx = xpu_ctx->x_context();

  const int max_batch_size = cu_seqlens_k.shape()[0] - 1;

  auto attn_mask_startend_row_indices = paddle::full({1, 1, kv_token_num, 2},
                                                     0,
                                                     paddle::DataType::INT32,
                                                     cu_seqlens_k.place());

  int r = fastdeploy::plugin::get_attn_mask_q(
      ctx,
      attn_mask_startend_row_indices.data<int>(),
      attn_mask_kv ? attn_mask_kv.get().data<int>() : nullptr,
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      kv_token_num,
      max_batch_size);
  PD_CHECK(r == 0, "fastdeploy::plugin::get_attn_mask_q failed.");

  return {attn_mask_startend_row_indices};
}

std::vector<std::vector<int64_t>> GetAttnMaskQInferShape(
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& cu_seqlens_k_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_kv_shape,
    const int kv_token_num) {
  return {{1, 1, kv_token_num, 2}};
}

std::vector<paddle::DataType> GetAttnMaskQInferDtype(
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& cu_seqlens_k_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_kv_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_STATIC_OP(get_attn_mask_q)
    .Inputs({"cu_seqlens_q",
             "cu_seqlens_k",
             paddle::Optional("attn_mask_offsets")})
    .Outputs({"attn_mask_q"})
    .Attrs({"kv_token_num: int"})
    .SetKernelFn(PD_KERNEL(GetAttnMaskQ))
    .SetInferShapeFn(PD_INFER_SHAPE(GetAttnMaskQInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetAttnMaskQInferDtype));
