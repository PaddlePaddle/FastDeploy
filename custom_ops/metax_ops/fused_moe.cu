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

#pragma once

#include "fused_moe_helper.h"
#include "helper.h"

namespace phi {

__global__ void compute_total_rows_before_expert_kernel(
    int* sorted_experts,
    const int64_t sorted_experts_len,
    const int64_t num_experts,
    int32_t* total_rows_before_expert) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;

  total_rows_before_expert[expert] =
      find_total_elts_leq_target(sorted_experts, sorted_experts_len, expert);
}

void compute_total_rows_before_expert(int* sorted_indices,
                                      const int64_t total_indices,
                                      const int64_t num_experts,
                                      int32_t* total_rows_before_expert,
                                      cudaStream_t stream) {
  const int threads = std::min(int64_t(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

}  // namespace phi

template <paddle::DataType T>
void FusedMoeKernel(const paddle::Tensor& input,
                    const paddle::Tensor& gate_weight,
                    const paddle::Tensor& up_gate_proj_weight,
                    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
                    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
                    const paddle::Tensor& down_proj_weight,
                    const paddle::optional<paddle::Tensor>& down_proj_scale,
                    const paddle::optional<paddle::Tensor>& down_proj_bias,
                    const std::string& quant_method,
                    const int moe_topk,
                    const bool group_moe,
                    const bool norm_topk_prob,
                    paddle::Tensor* output) {
  using namespace phi;
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto* output_data = output->data<data_t>();

  auto int8_moe_gemm_runner = McMoeGemmRunner<DataType_, int8_t>();

  auto moe_compute =
      McMoeHelper<data_t, DataType_>(quant_method, &int8_moe_gemm_runner);

  moe_compute.computeFFN(
      &input,
      &gate_weight,
      &up_gate_proj_weight,
      up_gate_proj_scale ? up_gate_proj_scale.get_ptr() : nullptr,
      up_gate_proj_bias ? up_gate_proj_bias.get_ptr() : nullptr,
      &down_proj_weight,
      down_proj_scale ? down_proj_scale.get_ptr() : nullptr,
      down_proj_bias ? down_proj_bias.get_ptr() : nullptr,
      nullptr,
      moe_topk,
      group_moe,
      norm_topk_prob,
      1.0,  // ComputeFFN
      "ffn",
      output);
}

std::vector<paddle::Tensor> FusedExpertMoe(
    const paddle::Tensor& input,
    const paddle::Tensor& gate_weight,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const std::string& quant_method,
    const int moe_topk,
    const bool norm_topk_prob,
    const bool group_moe) {
  const auto input_type = input.dtype();
  auto output = paddle::empty_like(input);

  if (output.dims()[0] == 0) {
    return {output};
  }

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      FusedMoeKernel<paddle::DataType::BFLOAT16>(input,
                                                 gate_weight,
                                                 up_gate_proj_weight,
                                                 up_gate_proj_scale,
                                                 up_gate_proj_bias,
                                                 down_proj_weight,
                                                 down_proj_scale,
                                                 down_proj_bias,
                                                 quant_method,
                                                 moe_topk,
                                                 group_moe,
                                                 norm_topk_prob,
                                                 &output);
      break;
    default:
      PD_THROW("Unsupported data type for FusedMoeKernel");
  }
  return {output};
}

std::vector<std::vector<int64_t>> FusedExpertMoeInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& gate_weight_shape,
    const std::vector<int64_t>& up_gate_proj_weight_shape,
    const std::vector<int64_t>& down_proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_scale_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>& down_proj_scale_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> FusedExpertMoeInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& gate_weight_dtype,
    const paddle::DataType& up_gate_proj_weight_dtype,
    const paddle::DataType& down_proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_bias_dtype,
    const paddle::optional<paddle::DataType>& up_gate_proj_scale_dtype,
    const paddle::optional<paddle::DataType>& down_proj_bias_dtype,
    const paddle::optional<paddle::DataType>& down_proj_scale_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(fused_expert_moe)
    .Inputs({"input",
             "gate_weight",
             "up_gate_proj_weight",
             "down_proj_weight",
             paddle::Optional("up_gate_proj_bias"),
             paddle::Optional("up_gate_proj_scale"),
             paddle::Optional("down_proj_bias"),
             paddle::Optional("down_proj_scale")})
    .Outputs({"output"})
    .Attrs({"quant_method:std::string",
            "moe_topk:int",
            "norm_topk_prob:bool",
            "group_moe:bool"})
    .SetKernelFn(PD_KERNEL(FusedExpertMoe))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedExpertMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedExpertMoeInferDtype));
