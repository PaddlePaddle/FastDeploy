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

#include "helper.h"
#include "iluvatar_context.h"

void __global__ restore_from_prefix_sum_kernel(const int64_t* prefix_sum,
                                               int64_t* tokens_per_expert,
                                               const int num_experts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_experts) {
    if (idx == 0) {
      tokens_per_expert[idx] = prefix_sum[idx];
    } else {
      tokens_per_expert[idx] = prefix_sum[idx] - prefix_sum[idx - 1];
    }
  }
}

std::vector<paddle::Tensor> RestoreTokensPerExpert(
    const paddle::Tensor& tokens_expert_prefix_sum) {
  const auto& prefix_sum_dims = tokens_expert_prefix_sum.dims();
  PADDLE_ENFORCE_EQ(prefix_sum_dims.size(),
                    1,
                    common::errors::InvalidArgument(
                        "tokens_expert_prefix_sum dims is [num_experts]"));

  const int num_experts = prefix_sum_dims[0];
  auto stream = tokens_expert_prefix_sum.stream();
  auto tokens_per_expert = GetEmptyTensor({num_experts},
                                          tokens_expert_prefix_sum.dtype(),
                                          tokens_expert_prefix_sum.place());

  const int block_size = 128;
  const int grid_size = (num_experts + block_size - 1) / block_size;
  restore_from_prefix_sum_kernel<<<grid_size, block_size, 0, stream>>>(
      const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
      tokens_per_expert.data<int64_t>(),
      num_experts);

  return {tokens_per_expert};
}

std::vector<std::vector<int64_t>> RestoreTokensPerExpertInferShape(
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape) {
  return {tokens_expert_prefix_sum_shape};
}

std::vector<paddle::DataType> RestoreTokensPerExpertInferDtype(
    const paddle::DataType& tokens_expert_prefix_sum_dtype) {
  return {tokens_expert_prefix_sum_dtype};
}

PD_BUILD_STATIC_OP(restore_tokens_per_expert)
    .Inputs({"tokens_expert_prefix_sum"})
    .Outputs({"tokens_per_expert"})
    .SetKernelFn(PD_KERNEL(RestoreTokensPerExpert))
    .SetInferShapeFn(PD_INFER_SHAPE(RestoreTokensPerExpertInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RestoreTokensPerExpertInferDtype));
