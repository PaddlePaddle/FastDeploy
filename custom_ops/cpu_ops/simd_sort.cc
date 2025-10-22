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
#include "x86simdsort-static-incl.h"

void probs_sort(const float *probs,
                int64_t *ProbsIds,
                float *ProbsVals,
                int vocab_size,
                int bsz) {
  float cursum = 0;
  std::vector<int64_t> elementsIds(vocab_size);
  std::vector<float> elementsProbs(vocab_size);
#pragma omp parallel for
  for (int j = 0; j < vocab_size; j++) {
    elementsIds[j] = j;
    elementsProbs[j] = probs[j];
  }
  x86simdsortStatic::keyvalue_qsort(
      elementsProbs.data(), elementsIds.data(), vocab_size, false, true);
#pragma omp parallel for
  for (int j = 0; j < vocab_size; ++j) {
    ProbsVals[j] = elementsProbs[j];
    ProbsIds[j] = elementsIds[j];
  }
}
std::vector<paddle::Tensor> SimdSort(const paddle::Tensor &probs) {
  const int bsz = probs.shape()[0];
  const int vocab_size = probs.shape()[1];
  auto sorted_indices =
      paddle::empty({bsz, vocab_size}, paddle::DataType::INT64, probs.place());
  auto sorted_probs = paddle::empty(
      {bsz, vocab_size}, paddle::DataType::FLOAT32, probs.place());
  probs_sort(probs.data<float>(),
             const_cast<int64_t *>(sorted_indices.data<int64_t>()),
             const_cast<float *>(sorted_probs.data<float>()),
             vocab_size,
             bsz);
  return {sorted_indices, sorted_probs};
}
std::vector<std::vector<int64_t>> SimdSortInferShape(
    const std::vector<int64_t> &probs_shape) {
  int64_t bsz = probs_shape[0];
  int64_t vocab_size = probs_shape[1];
  return {{bsz, vocab_size}, {bsz, vocab_size}};
}
std::vector<paddle::DataType> SimdSortInferDtype(
    const paddle::DataType &probs_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::FLOAT32};
}
PD_BUILD_STATIC_OP(simd_sort)
    .Inputs({"probs"})
    .Outputs({"sorted_indices_out", "sorted_probs_out"})
    .SetInferShapeFn(PD_INFER_SHAPE(SimdSortInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SimdSortInferDtype))
    .SetKernelFn(PD_KERNEL(SimdSort));
