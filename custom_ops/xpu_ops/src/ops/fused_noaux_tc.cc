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

std::vector<paddle::Tensor> FusedNoAuxTc(const paddle::Tensor& gating_logits,
                                         const paddle::Tensor& bias,
                                         const int n_group,
                                         const int topk_group,
                                         const int top_k,
                                         const bool apply_norm_weight,
                                         const float routed_scaling_factor) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  PD_CHECK(apply_norm_weight, "only support apply_norm_weight==true");

  auto gating_logits_dims = gating_logits.shape();
  int token_num = gating_logits_dims[0];
  int expert_num = gating_logits_dims[1];
  auto topk_idx = paddle::empty(
      {token_num, top_k}, paddle::DataType::INT32, gating_logits.place());
  auto topk_weights = paddle::empty(
      {token_num, top_k}, paddle::DataType::FLOAT32, gating_logits.place());
  int32_t* block_statistic = nullptr;
  if (token_num > 0) {
    int ret = infer_ops::moe_sigmoid_topk_norm_fusion(
        xpu_ctx->x_context(),
        gating_logits.data<float>(),
        const_cast<float*>(bias.data<float>()),
        routed_scaling_factor,
        topk_weights.mutable_data<float>(),
        topk_idx.mutable_data<int>(),
        block_statistic,
        token_num,
        expert_num,
        n_group,
        topk_group,
        top_k,
        0);
    PD_CHECK(ret == 0);
  }

  return {gating_logits,
          topk_weights,
          topk_idx};  // return gating_logits without change
}

std::vector<std::vector<int64_t>> FusedNoAuxTcInferShape(
    const std::vector<int64_t>& gating_logits_shape,
    const std::vector<int64_t>& bias_shape,
    const int n_group,
    const int topk_group,
    const int top_k,
    const bool apply_norm_weight,
    const float routed_scaling_factor) {
  std::vector<int64_t> topk_ids_shape = {gating_logits_shape[0], top_k};
  std::vector<int64_t> topk_weights_shape = {gating_logits_shape[0], top_k};
  return {gating_logits_shape, topk_weights_shape, topk_ids_shape};
}

std::vector<paddle::DataType> FusedNoAuxTcInferDtype(
    const paddle::DataType& gating_logits_dtype,
    const paddle::DataType& bias_dtype) {
  return {
      gating_logits_dtype, paddle::DataType::FLOAT32, paddle::DataType::INT32};
}

PD_BUILD_STATIC_OP(fused_noaux_tc)
    .Inputs({"gating_logits", "bias"})
    .Outputs({"gating_logits_out", "topk_weights", "topk_ids"})
    .Attrs({"n_group: int",
            "topk_group: int",
            "top_k: int",
            "apply_norm_weight: bool",
            "routed_scaling_factor: float"})
    .SetInplaceMap({{"gating_logits", "gating_logits_out"}})
    .SetKernelFn(PD_KERNEL(FusedNoAuxTc))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedNoAuxTcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedNoAuxTcInferDtype));
