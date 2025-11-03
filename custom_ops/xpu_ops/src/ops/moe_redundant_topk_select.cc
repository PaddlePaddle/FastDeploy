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

std::vector<paddle::Tensor> MoERedundantTopKSelect(
    const paddle::Tensor& gating_logits,
    const paddle::Tensor& expert_id_to_ep_rank_array,
    const paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,  // NOLINT
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  namespace api = baidu::xpu::api;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (gating_logits.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  PD_CHECK(apply_norm_weight, "only support apply_norm_weight==true");
  PD_CHECK(enable_softmax_top_k_fused,
           "only support enable_softmax_top_k_fused==true");
  PD_CHECK(bias.get_ptr() != nullptr, "only support bias != nullptr");

  auto gating_logits_dims = gating_logits.shape();
  int expert_num = gating_logits_dims[gating_logits_dims.size() - 1];
  int64_t token_num = 0;
  if (gating_logits_dims.size() == 3) {
    token_num = gating_logits_dims[0] * gating_logits_dims[1];
  } else {
    token_num = gating_logits_dims[0];
  }
  auto topk_ids = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::INT32, gating_logits.place());
  auto topk_ids_tmp = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::INT32, gating_logits.place());
  auto source_rows_tmp = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::INT32, gating_logits.place());
  auto topk_weights = paddle::empty(
      {token_num, moe_topk}, paddle::DataType::FLOAT32, gating_logits.place());

  const float* bias_data =
      bias.get_ptr() != nullptr ? bias.get_ptr()->data<float>() : nullptr;
  int ret = infer_ops::moe_redundant_softmax_topk_normed<float, float, int>(
      ctx,
      gating_logits.data<float>(),
      bias_data,
      expert_id_to_ep_rank_array.data<int>(),
      expert_in_rank_num_list.data<int>(),
      tokens_per_expert_stats_list.data<int>(),
      topk_weights.data<float>(),
      topk_ids.data<int>(),
      topk_ids_tmp.data<int>(),
      source_rows_tmp.data<int>(),
      expert_num,
      moe_topk,
      token_num,
      redundant_ep_rank_num_plus_one);
  PD_CHECK(ret == 0);

  return {topk_ids, topk_weights};
}

std::vector<std::vector<int64_t>> MoERedundantTopKSelectInferShape(
    const std::vector<int64_t>& gating_logits_shape,
    const std::vector<int64_t>& expert_id_to_ep_rank_array_shape,
    const std::vector<int64_t>& expert_in_rank_num_list_shape,
    const std::vector<int64_t>& tokens_per_expert_stats_list_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  int64_t token_rows = -1;
  if (gating_logits_shape.size() == 3) {
    token_rows = gating_logits_shape[0] * gating_logits_shape[1];
  } else {
    token_rows = gating_logits_shape[0];
  }

  std::vector<int64_t> topk_ids_shape = {token_rows, moe_topk};
  std::vector<int64_t> topk_weights_shape = {token_rows, moe_topk};
  return {topk_ids_shape, topk_weights_shape};
}

std::vector<paddle::DataType> MoERedundantTopKSelectInferDtype(
    const paddle::DataType& gating_logits_dtype,
    const paddle::DataType& expert_id_to_ep_rank_array_dtype,
    const paddle::DataType& expert_in_rank_num_list_dtype,
    const paddle::DataType& tokens_per_expert_stats_list_dtype,
    const paddle::optional<paddle::DataType>& bias_type,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one) {
  return {paddle::DataType::INT32, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(moe_redundant_topk_select)
    .Inputs({"gating_logits",
             "expert_id_to_ep_rank_array",
             "expert_in_rank_num_list",
             "tokens_per_expert_stats_list",
             paddle::Optional("bias")})
    .Outputs({"topk_ids", "topk_weights", "tokens_per_expert_stats_list_out"})
    .Attrs({"moe_topk: int",
            "apply_norm_weight: bool",
            "enable_softmax_top_k_fused:bool",
            "redundant_ep_rank_num_plus_one:int"})
    .SetInplaceMap({{"tokens_per_expert_stats_list",
                     "tokens_per_expert_stats_list_out"}})
    .SetKernelFn(PD_KERNEL(MoERedundantTopKSelect))
    .SetInferShapeFn(PD_INFER_SHAPE(MoERedundantTopKSelectInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoERedundantTopKSelectInferDtype));
