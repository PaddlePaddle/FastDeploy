// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "cutlass/numeric_conversion.h"
#include "group_swiglu_with_masked.h"
#include "helper.h"
#include "moe/fused_moe_helper.h"

template <typename DataT,
          typename NvType,
          typename WeightSavedT,
          cutlass::WintQuantMethod QuantMethod>
void WeightOnlyMoeFFNKernel(const paddle::Tensor& permute_input,
                            const paddle::Tensor& tokens_expert_prefix_sum,
                            const paddle::Tensor& up_gate_proj_weight,
                            const paddle::Tensor& down_proj_weight,
                            const paddle::Tensor* up_gate_proj_bias,
                            const paddle::Tensor* up_gate_proj_super_scale,
                            const paddle::Tensor* down_proj_super_scale,
                            const paddle::Tensor* up_gate_proj_local_scale,
                            const paddle::Tensor* up_gate_proj_code_scale,
                            const paddle::Tensor* up_gate_proj_code_zp,
                            const paddle::Tensor* down_proj_local_scale,
                            const paddle::Tensor* down_proj_code_scale,
                            const paddle::Tensor* down_proj_code_zp,
                            paddle::Tensor fc1_out,
                            paddle::Tensor ffn_out,
                            const int64_t total_rows_in_ll_else_minus1,
                            const int64_t actual_total_rows,
                            const int64_t inter_size,
                            const int64_t hidden_size,
                            const int num_experts,
                            bool used_in_ep_low_latency) {
  using namespace phi;
  using WeightOnlyTraits = cutlass::WintQuantTraits<NvType, QuantMethod>;
  using WeightType = typename WeightOnlyTraits::WeightType;

  typename WeightOnlyTraits::Arguments up_gate_proj_quant_args;
  typename WeightOnlyTraits::Arguments down_proj_quant_args;
  if constexpr (QuantMethod == cutlass::WintQuantMethod::kWeightOnlyInt2) {
    up_gate_proj_quant_args.local_scale_ptr =
        const_cast<uint8_t*>(up_gate_proj_local_scale->data<uint8_t>());
    up_gate_proj_quant_args.code_scale_ptr =
        const_cast<float*>(up_gate_proj_code_scale->data<float>());
    up_gate_proj_quant_args.code_zp_ptr =
        const_cast<float*>(up_gate_proj_code_zp->data<float>());

    down_proj_quant_args.local_scale_ptr =
        const_cast<uint8_t*>(down_proj_local_scale->data<uint8_t>());
    down_proj_quant_args.code_scale_ptr =
        const_cast<float*>(down_proj_code_scale->data<float>());
    down_proj_quant_args.code_zp_ptr =
        const_cast<float*>(down_proj_code_zp->data<float>());
  }

  auto moe_gemm_runner = MoeGemmRunner<NvType, WeightOnlyTraits>();
  auto stream = permute_input.stream();

  moe_gemm_runner.moe_gemm_bias_act(
      reinterpret_cast<const NvType*>(permute_input.data<DataT>()),
      reinterpret_cast<const WeightType*>(
          up_gate_proj_weight.data<WeightSavedT>()),
      reinterpret_cast<const NvType*>(
          up_gate_proj_super_scale ? up_gate_proj_super_scale->data<DataT>()
                                   : nullptr),
      reinterpret_cast<const NvType*>(
          up_gate_proj_bias ? up_gate_proj_bias->data<DataT>() : nullptr),
      reinterpret_cast<NvType*>(fc1_out.data<DataT>()),
      const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
      total_rows_in_ll_else_minus1,
      actual_total_rows,
      inter_size,
      hidden_size,
      num_experts,
      up_gate_proj_quant_args,
      "none",
      stream);

  paddle::Tensor act_out;
  if (used_in_ep_low_latency) {
    act_out = GroupSwigluWithMasked(fc1_out, tokens_expert_prefix_sum);
  } else {
    act_out = paddle::experimental::swiglu(fc1_out, nullptr);
  }

  moe_gemm_runner.moe_gemm(
      reinterpret_cast<const NvType*>(act_out.data<DataT>()),
      reinterpret_cast<const WeightType*>(
          down_proj_weight.data<WeightSavedT>()),
      reinterpret_cast<const NvType*>(down_proj_super_scale
                                          ? down_proj_super_scale->data<DataT>()
                                          : nullptr),
      reinterpret_cast<NvType*>(ffn_out.data<DataT>()),
      const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
      total_rows_in_ll_else_minus1,
      actual_total_rows,
      hidden_size,
      inter_size / 2,
      num_experts,
      down_proj_quant_args,
      stream);
}

template <paddle::DataType T>
void MoeFFNWint2Kernel(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_local_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_zp,
    const paddle::optional<paddle::Tensor>& down_proj_local_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_zp,
    paddle::Tensor ffn_out,
    bool used_in_ep_low_latency) {
  using namespace phi;
  using data_t = typename PDTraits<T>::data_t;
  using NvType = typename PDTraits<T>::DataType;

  auto place = permute_input.place();

  assert(permute_input.dims().size() == 3 || permute_input.dims().size() == 2);
  assert(up_gate_proj_weight.dims().size() == 3);

  const int num_experts = up_gate_proj_weight.dims()[0];
  const int hidden_size = permute_input.dims()[permute_input.dims().size() - 1];

  int inter_dim = up_gate_proj_weight.dims()[1] *
                  up_gate_proj_weight.dims()[2] / hidden_size;

  const int64_t inter_size = inter_dim * 4;

  int num_experts_ = num_experts;
  int num_max_tokens_per_expert = 0;
  int expanded_active_expert_rows = 0;

  paddle::Tensor fc1_out_tensor;
  if (permute_input.dims().size() == 3) {
    num_experts_ = permute_input.dims()[0];
    assert(num_experts == num_experts_);

    num_max_tokens_per_expert = permute_input.dims()[1];
    expanded_active_expert_rows = num_experts_ * num_max_tokens_per_expert;
    fc1_out_tensor = GetEmptyTensor(
        {num_experts_, num_max_tokens_per_expert, inter_size}, T, place);
  } else {
    expanded_active_expert_rows = permute_input.dims()[0];
    fc1_out_tensor =
        GetEmptyTensor({expanded_active_expert_rows, inter_size}, T, place);
  }

  // This is a trick.
  // expanded_active_expert_rows is not needed in variable group gemm.
  // but is needed in accommodating deepep low latency mode
  const int64_t total_rows_in_ll_else_minus1 =
      used_in_ep_low_latency ? expanded_active_expert_rows : -1;

  // When we tune the optimal configuration, we need the actual total_rows.
  const int64_t actual_total_rows = expanded_active_expert_rows;

  WeightOnlyMoeFFNKernel<data_t,
                         NvType,
                         uint8_t,
                         cutlass::WintQuantMethod::kWeightOnlyInt2>(
      permute_input,
      tokens_expert_prefix_sum,
      up_gate_proj_weight,
      down_proj_weight,
      const_cast<paddle::Tensor*>(up_gate_proj_bias.get_ptr()),
      const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr()),
      const_cast<paddle::Tensor*>(down_proj_scale.get_ptr()),
      const_cast<paddle::Tensor*>(up_gate_proj_local_scale.get_ptr()),
      const_cast<paddle::Tensor*>(up_gate_proj_code_scale.get_ptr()),
      const_cast<paddle::Tensor*>(up_gate_proj_code_zp.get_ptr()),
      const_cast<paddle::Tensor*>(down_proj_local_scale.get_ptr()),
      const_cast<paddle::Tensor*>(down_proj_code_scale.get_ptr()),
      const_cast<paddle::Tensor*>(down_proj_code_zp.get_ptr()),
      fc1_out_tensor,
      ffn_out,
      total_rows_in_ll_else_minus1,
      actual_total_rows,
      inter_size,
      hidden_size,
      num_experts,
      used_in_ep_low_latency);
}

paddle::Tensor MoeExpertFFNWint2Func(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_local_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_zp,
    const paddle::optional<paddle::Tensor>& down_proj_local_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_zp,
    const bool used_in_ep_low_latency) {
  const auto dtype = permute_input.dtype();
  auto ffn_out = paddle::empty_like(permute_input, dtype);

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      MoeFFNWint2Kernel<paddle::DataType::BFLOAT16>(permute_input,
                                                    tokens_expert_prefix_sum,
                                                    up_gate_proj_weight,
                                                    down_proj_weight,
                                                    up_gate_proj_bias,
                                                    up_gate_proj_scale,
                                                    down_proj_scale,
                                                    up_gate_proj_local_scale,
                                                    up_gate_proj_code_scale,
                                                    up_gate_proj_code_zp,
                                                    down_proj_local_scale,
                                                    down_proj_code_scale,
                                                    down_proj_code_zp,
                                                    ffn_out,
                                                    used_in_ep_low_latency);
      break;
    case paddle::DataType::FLOAT16:
      MoeFFNWint2Kernel<paddle::DataType::FLOAT16>(permute_input,
                                                   tokens_expert_prefix_sum,
                                                   up_gate_proj_weight,
                                                   down_proj_weight,
                                                   up_gate_proj_bias,
                                                   up_gate_proj_scale,
                                                   down_proj_scale,
                                                   up_gate_proj_local_scale,
                                                   up_gate_proj_code_scale,
                                                   up_gate_proj_code_zp,
                                                   down_proj_local_scale,
                                                   down_proj_code_scale,
                                                   down_proj_code_zp,
                                                   ffn_out,
                                                   used_in_ep_low_latency);
      break;
    default:
      PD_THROW("Unsupported data type for MoeExpertFFN");
  }
  return ffn_out;
}

std::vector<paddle::Tensor> MoeExpertFFNWint2(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_local_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_zp,
    const paddle::optional<paddle::Tensor>& down_proj_local_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_zp,
    const bool used_in_ep_low_latency) {
  return {MoeExpertFFNWint2Func(permute_input,
                                tokens_expert_prefix_sum,
                                up_gate_proj_weight,
                                down_proj_weight,
                                up_gate_proj_bias,
                                up_gate_proj_scale,
                                down_proj_scale,
                                up_gate_proj_local_scale,
                                up_gate_proj_code_scale,
                                up_gate_proj_code_zp,
                                down_proj_local_scale,
                                down_proj_code_scale,
                                down_proj_code_zp,
                                used_in_ep_low_latency)};
}

std::vector<std::vector<int64_t>> MoeExpertFFNWint2InferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& up_gate_proj_weight_shape,
    const std::vector<int64_t>& down_proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>&
        up_gate_proj_local_scale_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_code_scale_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_code_zp_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_local_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_code_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_code_zp_shape,
    const bool used_in_ep_low_latency) {
  return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNWint2InferDtype(
    const paddle::DataType& permute_input_dtype,
    const paddle::DataType& tokens_expert_prefix_sum_dtype,
    const paddle::DataType& up_gate_proj_weight_dtype,
    const paddle::DataType& down_proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_bias_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_local_scale_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_code_scale_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_code_zp_dtype,
    const paddle::optional<paddle::DataType>& down_proj_local_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_code_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_code_zp_dtype,
    const bool used_in_ep_low_latency) {
  return {permute_input_dtype};
}

/**
 * @brief Weight-Only Quantized Mixture of Experts (MoE) Feed-Forward Network
 * Operator
 *
 * This operator performs the expert computation in MoE architecture, including:
 * 1. First linear transformation (up_gate_proj) with optional quantization
 * 2. SwiGLU activation function
 * 3. Second linear transformation (down_proj) with optional quantization
 *
 * Supports multiple quantization methods including weight-only int4/int8 and
 * w4a8 quantization.
 *
 * Inputs:
 *   - permute_input: Permuted input tensor organized by expert
 *                   Shape: [total_tokens * top_k, hidden_size]
 *                   dtype: bfloat16/float16 (or int8 for w4a8)
 *   - tokens_expert_prefix_sum: Prefix sum array of token counts per expert for
 * group_gemm Shape: [num_experts] dtype: int64
 *   - up_gate_proj_weight: First FFN layer weights
 *                 Shape: [num_experts, inter_size * 2, hidden_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - down_proj_weight: Second FFN layer weights
 *                 Shape: [num_experts, hidden_size, inter_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - up_gate_proj_bias: Optional bias for first FFN layer
 *               Shape: [num_experts, inter_size * 2]
 *               dtype: Same as input
 *   - up_gate_proj_scale: Quantization scales for first FFN layer
 *                Shape: [num_experts, inter_size * 2]
 *                dtype: Same as input
 *   - down_proj_scale: Quantization scales for second FFN layer
 *                Shape: [num_experts, hidden_size]
 *                dtype: Same as input
 *
 * Outputs:
 *   - output_tensor: Output tensor after MoE FFN computation
 *                   Shape: Same as permute_input
 *                   dtype: Same as input (or up_gate_proj_scale dtype for w4a8)
 *
 * Attributes:
 *   - used_in_ep_low_latency: Whether running in low latency mode
 *                            Affects activation function implementation
 *
 * Note:
 * - Low latency mode uses specialized grouped SwiGLU implementation
 */
PD_BUILD_STATIC_OP(moe_expert_ffn_wint2)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "up_gate_proj_weight",
             "down_proj_weight",
             paddle::Optional("up_gate_proj_bias"),
             paddle::Optional("up_gate_proj_scale"),
             paddle::Optional("down_proj_scale"),
             paddle::Optional("up_gate_proj_local_scale"),
             paddle::Optional("up_gate_proj_code_scale"),
             paddle::Optional("up_gate_proj_code_zp"),
             paddle::Optional("down_proj_local_scale"),
             paddle::Optional("down_proj_code_scale"),
             paddle::Optional("down_proj_code_zp")})
    .Outputs({"output_tensor"})
    .Attrs({"used_in_ep_low_latency:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFNWint2))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNWint2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNWint2InferDtype));
