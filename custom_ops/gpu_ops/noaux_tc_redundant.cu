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

#include <algorithm>
#include <optional>

#include "helper.h"
#include "noauxtc_kernel.h"

std::vector<paddle::Tensor> NoauxTcRedundant(
    paddle::Tensor& scores,
    paddle::Tensor& scores_with_bias,
    paddle::Tensor& expert_id_to_ep_rank_array,
    paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,
    int n_group,
    int topk_group,
    int topk,
    bool renormalize,
    float routed_scaling_factor,
    int redundant_ep_rank_num_plus_one) {
  auto input_shape = scores_with_bias.shape();
  PD_CHECK(input_shape.size() == 2);
  int64_t num_tokens = input_shape[0];
  int64_t num_experts = input_shape[1];
  auto input_type = scores_with_bias.dtype();
  auto place = scores_with_bias.place();
  auto group_scores = paddle::empty({num_tokens, n_group}, input_type, place);
  auto topk_values = paddle::empty({num_tokens, topk}, input_type, place);
  auto topk_indices =
      paddle::empty({num_tokens, topk}, paddle::DataType::INT64, place);
  auto stream = scores_with_bias.stream();

  invokeNoAuxTcRedundant<float, int64_t>(
      reinterpret_cast<float*>(scores.data<float>()),
      reinterpret_cast<float*>(group_scores.data<float>()),
      reinterpret_cast<float*>(topk_values.data<float>()),
      reinterpret_cast<int64_t*>(topk_indices.data<int64_t>()),
      reinterpret_cast<float*>(scores_with_bias.data<float>()),
      reinterpret_cast<int*>(expert_id_to_ep_rank_array.data<int>()),
      reinterpret_cast<int*>(expert_in_rank_num_list.data<int>()),
      reinterpret_cast<int*>(tokens_per_expert_stats_list.data<int>()),
      num_tokens,
      num_experts,
      n_group,
      topk_group,
      topk,
      renormalize,
      routed_scaling_factor,
      redundant_ep_rank_num_plus_one,
      stream);

  return {scores, topk_values, topk_indices};
}

std::vector<paddle::DataType> NoauxTcRedundantInferDtype(
    const paddle::DataType& scores_dtype,
    const paddle::DataType& scores_with_bias_dtype) {
  return {scores_dtype, scores_dtype, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> NoauxTcRedundantInferShape(
    const std::vector<int64_t>& scores_shape,
    const std::vector<int64_t>&,
    const int topk) {
  auto num_tokens = scores_shape[0];
  auto topk_values_shape = std::vector<int64_t>{num_tokens, topk};
  auto topk_indices_shape = std::vector<int64_t>{num_tokens, topk};
  return {scores_shape, topk_values_shape, topk_indices_shape};
}

PD_BUILD_STATIC_OP(noaux_tc_redundant)
    .Inputs({"scores",
             "scores_with_bias",
             "expert_id_to_ep_rank_array",
             "expert_in_rank_num_list",
             "tokens_per_expert_stats_list"})
    .Outputs({"output_tensor",
              "topk_values",
              "topk_indices",
              "tokens_per_expert_stats_list_out"})
    .Attrs({"n_group: int",
            "topk_group: int",
            "topk:int",
            "renormalize: bool",
            "routed_scaling_factor: float",
            "redundant_ep_rank_num_plus_one:int"})
    .SetInplaceMap({{"tokens_per_expert_stats_list",
                     "tokens_per_expert_stats_list_out"}})
    .SetKernelFn(PD_KERNEL(NoauxTcRedundant))
    .SetInferShapeFn(PD_INFER_SHAPE(NoauxTcRedundantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(NoauxTcRedundantInferDtype));
