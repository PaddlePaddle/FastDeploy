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

#include <infer_ops.h>
#include <xft_api.h>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

std::vector<paddle::Tensor> MoeTopkSelect(
    const paddle::Tensor& gating_logits,
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  PD_CHECK(apply_norm_weight, "only support apply_norm_weight==true");

  auto gating_logits_dims = gating_logits.shape();
  int token_num = gating_logits_dims[0];
  int expert_num = gating_logits_dims[1];
  auto topk_ids = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::INT32, gating_logits.place());
  auto topk_weights = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::FLOAT32, gating_logits.place());
  int32_t* block_statistic = nullptr;
  const float* bias_data =
      bias.get_ptr() != nullptr ? bias.get_ptr()->data<float>() : nullptr;
  if (token_num > 0) {
    int ret = infer_ops::moe_softmax_topk_norm_fusion(
        xpu_ctx->x_context(),
        gating_logits.data<float>(),
        topk_weights.mutable_data<float>(),
        topk_ids.mutable_data<int>(),
        block_statistic,
        token_num,
        expert_num,
        moe_topk,
        0,
        bias_data);
    PD_CHECK(ret == 0);
  }

  return {topk_ids, topk_weights};
}

std::vector<std::vector<int64_t>> MoeTopkSelectInferShape(
    const std::vector<int64_t>& gating_logits_shape,
    const std::vector<int64_t>& bias_shape,
    const int moe_topk,
    const bool apply_norm_weight) {
  std::vector<int64_t> topk_ids_shape = {gating_logits_shape[0], moe_topk};
  std::vector<int64_t> topk_weights_shape = {gating_logits_shape[0], moe_topk};
  return {topk_ids_shape, topk_weights_shape};
}

std::vector<paddle::DataType> MoeTopkSelectInferDtype(
    const paddle::DataType& gating_logits_dtype,
    const paddle::DataType& bias_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::FLOAT32};
}

PD_BUILD_STATIC_OP(moe_topk_select)
    .Inputs({"gating_logits", paddle::Optional("bias")})
    .Outputs({"topk_ids", "topk_weights"})
    .Attrs({"moe_topk: int", "apply_norm_weight: bool"})
    .SetKernelFn(PD_KERNEL(MoeTopkSelect))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeTopkSelectInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeTopkSelectInferDtype));
