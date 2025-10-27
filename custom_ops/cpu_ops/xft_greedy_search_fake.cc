// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstdio>
#include <iostream>
#include "paddle/extension.h"

std::vector<paddle::Tensor> XftGreedySearch(const paddle::Tensor &probs) {
  const int bsz = probs.shape()[0];
  const int vocab_size = probs.shape()[1];
  auto next_tokens =
      paddle::empty({bsz, 1}, paddle::DataType::INT64, probs.place());
  return {next_tokens};
}
std::vector<std::vector<int64_t>> XftGreedySearchInferShape(
    const std::vector<int64_t> &probs_shape) {
  int64_t bsz = probs_shape[0];
  return {{bsz, 1}};
}
std::vector<paddle::DataType> XftGreedySearchInferDtype(
    const paddle::DataType &probs_dtype) {
  return {paddle::DataType::INT64};
}
PD_BUILD_STATIC_OP(xft_greedy_search)
    .Inputs({"probs"})
    .Outputs({"next_tokens_ids"})
    .SetInferShapeFn(PD_INFER_SHAPE(XftGreedySearchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(XftGreedySearchInferDtype))
    .SetKernelFn(PD_KERNEL(XftGreedySearch));
