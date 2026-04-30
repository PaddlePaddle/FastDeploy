
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

#include <algorithm>
#include <optional>

#include "helper.h"
#include "noauxtc_kernel.h"

std::vector<paddle::Tensor> NoauxTc(paddle::Tensor& gating_output,
                                    paddle::Tensor& e_score_correction_bias,
                                    int n_group,
                                    int topk_group,
                                    int topk,
                                    bool renormalize,
                                    float routed_scaling_factor) {
  auto input_shape = gating_output.shape();
  PD_CHECK(input_shape.size() == 2);
  int64_t num_tokens = input_shape[0];
  int64_t num_experts = input_shape[1];
  auto input_type = gating_output.dtype();
  auto place = gating_output.place();
  // scores: sparse output written by the kernel (unbiased sigmoid, topk
  // positions filled)
  auto scores = paddle::empty({num_tokens, num_experts}, input_type, place);
  // group_scores: warp-local register in _dev kernel, no longer a global
  // buffer; still allocated here as a placeholder passed to invokeNoAuxTc
  // (unused by kernel).
  auto group_scores = paddle::empty({num_tokens, n_group}, input_type, place);
  auto topk_values = paddle::empty({num_tokens, topk}, input_type, place);
  auto topk_indices =
      paddle::empty({num_tokens, topk}, paddle::DataType::INT64, place);
  auto stream = gating_output.stream();

  invokeNoAuxTc<float, int64_t>(
      reinterpret_cast<float*>(gating_output.data<float>()),
      reinterpret_cast<float*>(e_score_correction_bias.data<float>()),
      reinterpret_cast<float*>(scores.data<float>()),
      reinterpret_cast<float*>(group_scores.data<float>()),
      reinterpret_cast<float*>(topk_values.data<float>()),
      reinterpret_cast<int64_t*>(topk_indices.data<int64_t>()),
      num_tokens,
      num_experts,
      n_group,
      topk_group,
      topk,
      renormalize,
      routed_scaling_factor,
      stream);

  return {scores, topk_values, topk_indices};
}

std::vector<paddle::DataType> NoauxTcInferDtype(
    const paddle::DataType& gating_output_dtype,
    const paddle::DataType& bias_dtype) {
  return {gating_output_dtype, gating_output_dtype, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> NoauxTcInferShape(
    const std::vector<int64_t>& gating_output_shape,
    const std::vector<int64_t>&,
    const int topk) {
  auto num_tokens = gating_output_shape[0];
  auto num_experts = gating_output_shape[1];
  auto scores_shape = std::vector<int64_t>{num_tokens, num_experts};
  auto topk_values_shape = std::vector<int64_t>{num_tokens, topk};
  auto topk_indices_shape = std::vector<int64_t>{num_tokens, topk};
  return {scores_shape, topk_values_shape, topk_indices_shape};
}

PD_BUILD_STATIC_OP(noaux_tc)
    .Inputs({"gating_output", "e_score_correction_bias"})
    .Outputs({"output_tensor", "topk_values", "topk_indices"})
    .Attrs({"n_group: int",
            "topk_group: int",
            "topk:int",
            "renormalize: bool",
            "routed_scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(NoauxTc))
    .SetInferShapeFn(PD_INFER_SHAPE(NoauxTcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(NoauxTcInferDtype));
