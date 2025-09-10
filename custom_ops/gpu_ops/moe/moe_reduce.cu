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

// Ignore CUTLASS warnings about type punning

#pragma once

#include "helper.h"
#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_op.h"

template <paddle::DataType T>
void MoeReduceKernel(const paddle::Tensor &ffn_out,
                     const paddle::Tensor &top_k_weight,
                     const paddle::Tensor &permute_indices_per_token,
                     const paddle::Tensor &top_k_indices,
                     const paddle::optional<paddle::Tensor> &down_proj_bias,
                     const bool norm_topk_prob,
                     const float routed_scaling_factor, const int num_rows,
                     const int hidden_size, const int topk,
                     paddle::Tensor *output) {
  using namespace phi;
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto stream = ffn_out.stream();

  finalize_moe_routing_kernelLauncher(
      ffn_out.data<data_t>(), output->data<data_t>(),
      down_proj_bias ? down_proj_bias->data<data_t>() : nullptr,
      top_k_weight.data<float>(), permute_indices_per_token.data<int32_t>(),
      top_k_indices.data<int>(), num_rows, hidden_size, topk,
      static_cast<int>(1), norm_topk_prob, routed_scaling_factor, stream);
}

paddle::Tensor MoeExpertReduceFunc(
    const paddle::Tensor &ffn_out, const paddle::Tensor &top_k_weight,
    const paddle::Tensor &permute_indices_per_token,
    const paddle::Tensor &top_k_indices,
    const paddle::optional<paddle::Tensor> &down_proj_bias,
    const bool norm_topk_prob, const float routed_scaling_factor) {
  const auto input_type = ffn_out.dtype();
  auto place = ffn_out.place();

  const int topk = top_k_indices.dims()[1];
  const int num_rows = ffn_out.dims()[0] / topk;
  const int hidden_size = ffn_out.dims()[1];

  auto output = GetEmptyTensor({num_rows, hidden_size}, input_type, place);

  if(num_rows == 0){
    return output;
  }

  switch (input_type) {
  case paddle::DataType::BFLOAT16:
    MoeReduceKernel<paddle::DataType::BFLOAT16>(
        ffn_out, top_k_weight, permute_indices_per_token, top_k_indices,
        down_proj_bias, norm_topk_prob, routed_scaling_factor, num_rows, hidden_size,
        topk, &output);
    break;
  case paddle::DataType::FLOAT16:
    MoeReduceKernel<paddle::DataType::BFLOAT16>(
        ffn_out, top_k_weight, permute_indices_per_token, top_k_indices,
        down_proj_bias, norm_topk_prob, routed_scaling_factor, num_rows, hidden_size,
        topk, &output);
    break;
  default:
    PD_THROW("Unsupported data type for MoeDispatchKernel");
  }
  return output;
}

std::vector<paddle::Tensor>
MoeExpertReduce(const paddle::Tensor &ffn_out,
                const paddle::Tensor &top_k_weight,
                const paddle::Tensor &permute_indices_per_token,
                const paddle::Tensor &top_k_indices,
                const paddle::optional<paddle::Tensor> &down_proj_bias,
                const bool norm_topk_prob, const float routed_scaling_factor) {
  return {MoeExpertReduceFunc(ffn_out, top_k_weight, permute_indices_per_token,
                              top_k_indices, down_proj_bias, norm_topk_prob,
                              routed_scaling_factor)};
}

std::vector<std::vector<int64_t>> MoeExpertReduceInferShape(
    const std::vector<int64_t> &ffn_out_shape,
    const std::vector<int64_t> &top_k_weight_shape,
    const std::vector<int64_t> &permute_indices_per_token_shape,
    const std::vector<int64_t> &top_k_indices_shape,
    const paddle::optional<std::vector<int64_t>> &down_proj_bias_shape) {
  const int moe_topk = top_k_indices_shape[1];
  auto out_shape = ffn_out_shape;
  if (out_shape[0] != -1) out_shape[0] /= moe_topk;
  return {out_shape};
}

std::vector<paddle::DataType> MoeExpertReduceInferDtype(
    const paddle::DataType &ffn_out_dtype,
    const paddle::DataType &top_k_weight_dtype,
    const paddle::DataType &permute_indices_per_token_dtype,
    const paddle::DataType &top_k_indices_dtype,
    const paddle::optional<paddle::DataType> &down_proj_bias_dtype) {
  return {ffn_out_dtype};
}


/**
 * @brief Mixture of Experts (MoE) Expert Reduce Operator
 *
 * This operator performs the following key functions:
 * 1. Combines outputs from multiple experts based on routing weights
 * 2. Applies optional bias and scaling to the combined output
 * 3. Restores the original token order from permuted expert outputs
 *
 * Inputs:
 *   - ffn_out: Outputs from all expert networks (permuted)
 *             Shape: [total_tokens * moe_topk, hidden_size]
 *             dtype: bfloat16 or float16
 *   - top_k_weight: Routing weights for top-k experts per token
 *                  Shape: [total_tokens, moe_topk]
 *                  dtype: float32
 *   - permute_indices_per_token: Indices mapping for reconstructing original order
 *                               Shape: [moe_topk, total_tokens]
 *                               dtype: int32
 *   - top_k_indices: Indices of selected top-k experts for each token
 *                   Shape: [total_tokens, moe_topk]
 *                   dtype: int32
 *   - down_proj_bias: Optional bias term for expert outputs (hidden_size)
 *
 * Outputs:
 *   - output: Combined expert outputs in original token order
 *            Shape: [total_tokens, hidden_size]
 *            dtype: Same as ffn_out
 *
 * Attributes:
 *   - norm_topk_prob: Whether to normalize top-k probabilities
 *                    (true: weights sum to 1 for each token,
 *                     false: use raw weights)
 *   - routed_scaling_factor: Scaling factor applied to top-k probabilities
 *
 * Note:
 * - The operator expects permuted expert outputs from moe_expert_dispatch
 * - When norm_topk_prob is true, weights are normalized per token
 * - The routed_scaling_factor is typically used to balance expert contributions
 * - For optimal performance, hidden_size should be a multiple of 128
 */
PD_BUILD_STATIC_OP(moe_expert_reduce)
    .Inputs({"ffn_out", "top_k_weight", "permute_indices_per_token",
             "top_k_indices", paddle::Optional("down_proj_bias")})
    .Outputs({"output"})
    .Attrs({"norm_topk_prob:bool", "routed_scaling_factor:float"})
    .SetKernelFn(PD_KERNEL(MoeExpertReduce))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertReduceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertReduceInferDtype));
