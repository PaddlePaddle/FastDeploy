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
void moe_align_block_size(const paddle::Tensor& topk_ids,
                          int64_t num_experts,
                          int64_t block_size,
                          paddle::Tensor& sorted_token_ids,
                          paddle::Tensor& experts_ids,
                          paddle::Tensor& num_tokens_post_pad,
                          paddle::Tensor& cumsum_buffer,
                          bool pad_sorted_token_ids);

std::vector<std::vector<int64_t>> tritonmoe_preprocessInferShape(
    const std::vector<int64_t>& topk_ids,
    int64_t num_experts,
    int64_t GEMM_BLOCK_SIZE_M) {
  int topk_ids_numel = 1;
  for (int64_t dim : topk_ids) {
    topk_ids_numel *= static_cast<int>(dim);
  }
  int max_num_tokens_padded;
  if (topk_ids_numel < num_experts + 1) {
    max_num_tokens_padded = topk_ids_numel * GEMM_BLOCK_SIZE_M;
  } else {
    max_num_tokens_padded =
        topk_ids_numel + (num_experts + 1) * (GEMM_BLOCK_SIZE_M - 1);
  }

  std::vector<int64_t> sorted_ids = {max_num_tokens_padded};

  int max_num_m_blocks =
      (max_num_tokens_padded + GEMM_BLOCK_SIZE_M - 1) / GEMM_BLOCK_SIZE_M;
  std::vector<int64_t> experts_ids = {max_num_m_blocks};
  std::vector<int64_t> num_tokens_post_pad = {1};

  return {sorted_ids, experts_ids, num_tokens_post_pad};
}

std::vector<paddle::DataType> tritonmoe_preprocessIferDtype(
    const paddle::DataType& topk_ids,
    int64_t num_experts,
    int64_t GEMM_BLOCK_SIZE_M) {
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

/*
supporse num_experts = 8, GEMM_BLOCK_SIZE_M = 4,
topk_ids.shape = [4,4], means=topk=4
topk_ids=
[7 6 5 4
1 2 3 4
0 1 2 3
0 3 2 1]

Then return value `sorted_ids` is
8,12,16,16
4,9,15,16
5,10,14,16
6,11,13,16
3,7,16,16
2,16,16,16
1,16,16,16
0,16,16,16
*/

std::vector<paddle::Tensor> tritonmoe_preprocess_kernel(
    const paddle::Tensor& topk_ids,
    int64_t num_experts,
    int64_t GEMM_BLOCK_SIZE_M) {
  int topk_ids_numel = static_cast<int>(topk_ids.numel());

  int max_num_tokens_padded;
  if (topk_ids_numel < num_experts + 1) {
    max_num_tokens_padded = topk_ids_numel * GEMM_BLOCK_SIZE_M;
  } else {
    max_num_tokens_padded =
        topk_ids_numel + (num_experts + 1) * (GEMM_BLOCK_SIZE_M - 1);
  }

  auto sorted_ids = paddle::full({max_num_tokens_padded},
                                 topk_ids_numel,
                                 paddle::DataType::INT32,
                                 topk_ids.place());

  int max_num_m_blocks =
      (max_num_tokens_padded + GEMM_BLOCK_SIZE_M - 1) / GEMM_BLOCK_SIZE_M;

  auto experts_ids = paddle::empty(
      {max_num_m_blocks}, paddle::DataType::INT32, topk_ids.place());

  auto num_tokens_post_pad =
      paddle::empty({1}, paddle::DataType::INT32, topk_ids.place());

  auto cumsum_buffer = paddle::zeros(
      {num_experts + 2}, paddle::DataType::INT32, topk_ids.place());

  using scalar_t = int64_t;
  moe_align_block_size<scalar_t>(topk_ids,
                                 num_experts + 1,
                                 GEMM_BLOCK_SIZE_M,
                                 sorted_ids,
                                 experts_ids,
                                 num_tokens_post_pad,
                                 cumsum_buffer,
                                 true);

  return {sorted_ids, experts_ids, num_tokens_post_pad};
}

PD_BUILD_STATIC_OP(tritonmoe_preprocess)
    .Inputs({"topk_ids"})
    .Attrs({"num_experts: int64_t", "GEMM_BLOCK_SIZE_M: int64_t"})
    .Outputs({"sorted_ids", "experts_ids", "num_tokens_post_pad"})
    .SetKernelFn(PD_KERNEL(tritonmoe_preprocess_kernel))
    .SetInferShapeFn(PD_INFER_SHAPE(tritonmoe_preprocessInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(tritonmoe_preprocessIferDtype));
