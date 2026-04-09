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

#include "helper.h"
#include "paddle/extension.h"

template <typename scalar_t, bool kComputeCumsum>
__global__ void cuda_kernel(const scalar_t *__restrict__ topk_ids,
                            int32_t *__restrict__ res,
                            int32_t *__restrict__ res_padded,
                            int32_t *__restrict__ res_padded_cumsum,
                            size_t numel,
                            int num_experts) {
  extern __shared__ int32_t tokens_per_ep[];

  for (size_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_per_ep[i] = 0;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= 0) atomicAdd(&tokens_per_ep[expert_id], 1);
  }

  __syncthreads();

  if constexpr (kComputeCumsum) {
    if (threadIdx.x == 0) {
      int32_t running_sum = 0;
      for (int i = 0; i < num_experts; i++) {
        int32_t count = tokens_per_ep[i];
        int32_t padded = (count + 127) / 128 * 128;
        res[i] = count;
        res_padded[i] = padded;
        running_sum += padded;
        res_padded_cumsum[i] = running_sum;
      }
    }
  } else {
    for (size_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
      res[i] = tokens_per_ep[i];
      res_padded[i] = (tokens_per_ep[i] + 127) / 128 * 128;
    }
  }
}

std::vector<paddle::Tensor> count_tokens_per_expert_func(
    const paddle::Tensor &topk_ids,
    int64_t num_experts,
    bool compute_padded_cumsum) {
  int topk_ids_numel = topk_ids.shape()[0] * topk_ids.shape()[1];

  int64_t num_rows = compute_padded_cumsum ? 3 : 2;
  auto token_nums_per_expert = paddle::empty(
      {num_rows, num_experts}, paddle::DataType::INT32, topk_ids.place());

  auto stream = topk_ids.stream();
  using scalar_t = int64_t;

  if (compute_padded_cumsum) {
    cuda_kernel<scalar_t, true>
        <<<1, 1024, num_experts * sizeof(int32_t), stream>>>(
            topk_ids.data<scalar_t>(),
            token_nums_per_expert.data<int32_t>(),
            token_nums_per_expert.data<int32_t>() + num_experts,
            token_nums_per_expert.data<int32_t>() + 2 * num_experts,
            topk_ids_numel,
            num_experts);
  } else {
    cuda_kernel<scalar_t, false>
        <<<1, 1024, num_experts * sizeof(int32_t), stream>>>(
            topk_ids.data<scalar_t>(),
            token_nums_per_expert.data<int32_t>(),
            token_nums_per_expert.data<int32_t>() + num_experts,
            nullptr,
            topk_ids_numel,
            num_experts);
  }

  return {token_nums_per_expert};
}

std::vector<paddle::DataType> count_tokens_per_expert_func_infer_dtype(
    const paddle::DataType &topk_ids_dtype,
    int64_t num_experts,
    bool compute_padded_cumsum) {
  return {paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> count_tokens_per_expert_func_infer_shape(
    const std::vector<int64_t> &topk_ids_shape,
    int64_t num_experts,
    bool compute_padded_cumsum) {
  int64_t num_rows = compute_padded_cumsum ? 3 : 2;
  return {{num_rows, num_experts}};
}

PD_BUILD_STATIC_OP(count_tokens_per_expert_func)
    .Inputs({"topk_ids"})
    .Outputs({"token_nums_per_expert"})
    .Attrs({"num_experts:int64_t", "compute_padded_cumsum:bool"})
    .SetKernelFn(PD_KERNEL(count_tokens_per_expert_func))
    .SetInferShapeFn(PD_INFER_SHAPE(count_tokens_per_expert_func_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(count_tokens_per_expert_func_infer_dtype));
