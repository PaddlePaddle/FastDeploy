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
#include "fused_moe_helper.h"
#include "helper.h"

template <paddle::DataType T>
void MoeFFNKernel(paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& up_gate_proj_weight,
                  const paddle::Tensor& down_proj_weight,
                  const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
                  const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
                  const paddle::optional<paddle::Tensor>& down_proj_scale,
                  const paddle::optional<paddle::Tensor>& expert_idx_per_token,
                  const std::string& quant_method) {
  using namespace phi;
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto permuted_input_ptr = permute_input.data<data_t>();
  auto place = permute_input.place();
  auto input_type = permute_input.dtype();
  auto stream = permute_input.stream();

  auto int8_moe_gemm_runner = McMoeGemmRunner<DataType_, int8_t>();

  const int expanded_active_expert_rows =
      permute_input.dims()[0];  // permute_input.dims(): m, k
  const int num_experts = up_gate_proj_weight.dims()[0];  // batchsize
  const int hidden_size = up_gate_proj_weight.dims()[2];  // n
  int inter_dim = up_gate_proj_weight.dims()[1];          // k

  const int64_t inter_size = inter_dim;  // since weight_only_int_8
  paddle::Tensor fc1_out_tensor = GetEmptyTensor(
      {expanded_active_expert_rows, inter_size}, input_type, place);
  auto fc1_out_ptr = fc1_out_tensor.data<data_t>();

  mctlassExOrder_t row_major = mctlassExOrder_t::MCTLASS_EX_ORDER_ROW_MAJOR;
  mctlassExOrder_t column_major =
      mctlassExOrder_t::MCTLASS_EX_ORDER_COLUMN_MAJOR;
  auto m_num_tile =
      GetEmptyTensor({num_experts}, paddle::DataType::INT32, place);
  int* m_num_tile_ptr = reinterpret_cast<int*>(m_num_tile.data<int>());

  // ffn1
  auto fc1_expert_biases =
      up_gate_proj_bias
          ? const_cast<paddle::Tensor*>(up_gate_proj_bias.get_ptr())
                ->data<data_t>()
          : nullptr;
  auto fc1_expert_scales =
      const_cast<paddle::Tensor*>(up_gate_proj_scale.get_ptr())->data<data_t>();
  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.mc_grouped_gemm_basic_kernel(
        reinterpret_cast<const DataType_*>(permuted_input_ptr),
        row_major,
        reinterpret_cast<const int8_t*>(up_gate_proj_weight.data<int8_t>()),
        column_major,
        reinterpret_cast<const DataType_*>(fc1_expert_scales),
        reinterpret_cast<const DataType_*>(fc1_expert_biases),
        reinterpret_cast<DataType_*>(fc1_out_ptr),
        row_major,
        tokens_expert_prefix_sum.data<int>(),
        m_num_tile_ptr,
        num_experts,
        expanded_active_expert_rows,
        inter_dim,
        hidden_size,
        stream);
  } else {
    throw std::runtime_error("Unsupported gemm method: " + quant_method);
  }

  // swiglu
  auto act_out_tensor = paddle::experimental::swiglu(fc1_out_tensor, nullptr);
  auto act_out = act_out_tensor.data<data_t>();

  auto fc2_expert_scales =
      const_cast<paddle::Tensor*>(down_proj_scale.get_ptr())->data<data_t>();

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.mc_grouped_gemm_basic_kernel(
        reinterpret_cast<const DataType_*>(act_out),
        row_major,
        reinterpret_cast<const int8_t*>(down_proj_weight.data<int8_t>()),
        column_major,
        reinterpret_cast<const DataType_*>(fc2_expert_scales),
        nullptr,
        reinterpret_cast<DataType_*>(permuted_input_ptr),
        row_major,
        tokens_expert_prefix_sum.data<int>(),
        m_num_tile_ptr,
        num_experts,
        expanded_active_expert_rows,
        hidden_size,
        inter_dim / 2,
        stream);
  } else {
    throw std::runtime_error("Unsupported gemm method: " + quant_method);
  }
}

std::vector<paddle::Tensor> MoeExpertFFN(
    paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method) {
  assert(quant_method == "weight_only_int8");
  const auto input_type = permute_input.dtype();

  if (permute_input.numel() == 0) {
    return {permute_input};
  }

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeFFNKernel<paddle::DataType::BFLOAT16>(permute_input,
                                               tokens_expert_prefix_sum,
                                               up_gate_proj_weight,
                                               down_proj_weight,
                                               up_gate_proj_bias,
                                               up_gate_proj_scale,
                                               down_proj_scale,
                                               expert_idx_per_token,
                                               quant_method);
      break;
    default:
      PD_THROW("Unsupported data type for MoeFFNhKernel");
  }
  return {permute_input};
}

std::vector<std::vector<int64_t>> MoeExpertFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& up_gate_proj_weight_shape,
    const std::vector<int64_t>& down_proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& expert_idx_per_token_shape,
    const std::string& quant_method) {
  return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNInferDtype(
    const paddle::DataType& permute_input_dtype,
    const paddle::DataType& tokens_expert_prefix_sum_dtype,
    const paddle::DataType& up_gate_proj_weight_dtype,
    const paddle::DataType& down_proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_bias_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& expert_idx_per_token_dtype) {
  return {permute_input_dtype};
}

PD_BUILD_OP(moe_expert_ffn)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "up_gate_proj_weight",
             "down_proj_weight",
             paddle::Optional("up_gate_proj_bias"),
             paddle::Optional("up_gate_proj_scale"),
             paddle::Optional("down_proj_scale"),
             paddle::Optional("expert_idx_per_token")})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNInferDtype));
