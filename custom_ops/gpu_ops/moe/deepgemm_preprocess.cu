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

template <typename scalar_t>
__global__ void
cuda_kernel(const scalar_t *__restrict__ topk_ids, int32_t *__restrict__ res,
            int32_t *__restrict__ res_padded, size_t numel, int num_experts) {

  extern __shared__ int32_t tokens_per_ep[];

  for (size_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_per_ep[i] = 0;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
    int32_t expert_id = topk_ids[i];
    if(expert_id >= 0) atomicAdd(&tokens_per_ep[expert_id], 1);
  }

  __syncthreads();

  for (size_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    res[i] = tokens_per_ep[i];
    res_padded[i] = (res[i] + 127) / 128 * 128;
  }
}

paddle::Tensor count_tokens_per_expert_func(const paddle::Tensor &topk_ids,
                                            int64_t num_experts) {

  int topk_ids_numel = topk_ids.shape()[0] * topk_ids.shape()[1];

  auto token_nums_per_expert = paddle::empty(
      {2, num_experts}, paddle::DataType::INT32, topk_ids.place());

  auto stream = topk_ids.stream();
  using scalar_t = int64_t;

  cuda_kernel<<<1, 1024, num_experts * sizeof(int32_t), stream>>>(
      topk_ids.data<scalar_t>(), token_nums_per_expert.data<int32_t>(),
      token_nums_per_expert.data<int32_t>() + num_experts, topk_ids_numel,
      num_experts);
  return token_nums_per_expert;
}
