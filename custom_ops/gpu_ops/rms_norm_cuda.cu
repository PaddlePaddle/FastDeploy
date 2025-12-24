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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include <cassert>
#include <vector>

#include "rms_norm_cuda.h"  // NOLINT
#include "paddle/extension.h"

#ifdef CUSTOM_OP_WITH_SPMD
#include "paddle/phi/api/ext/spmd_infer.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

#define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int *p_rows,
                        int *p_cols) {
  int rows = 1;
  for (int i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int cols = shape[shape.size() - 1];
  *p_rows = rows;
  *p_cols = cols;
}

std::vector<paddle::Tensor> RMSLnFwd(const paddle::Tensor &x,
                                     const paddle::Tensor &scale,
                                     float epsilon) {
  const auto &scale_shape = scale.shape();
  const auto &x_shape = x.shape();
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto place = x.place();
  
  auto y = paddle::empty(x_shape, scale.type(), place);
  auto variance_shape = x_shape;
  variance_shape.pop_back();
  auto invvar = paddle::empty(variance_shape, paddle::DataType::FLOAT32, place);
  cuda_rms_norm(x, scale, rows, cols, epsilon, &y, &invvar);
  return {y, invvar};
}


std::vector<std::vector<int64_t>> RMSLnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    float epsilon) {
  auto variance_shape = x_shape;
  variance_shape.pop_back();
  return {x_shape, variance_shape};
}

std::vector<paddle::DataType> RMSLnFwdInferDtype(paddle::DataType x_dtype,
                                                 paddle::DataType scale_dtype) {
  return {x_dtype, paddle::DataType::FLOAT32};
}

// PD_BUILD_OP(fused_rms_norm_infer)
//     .Inputs({"x", "scale"})
//     .Outputs({"y", "invvar"})
//     .Attrs({"epsilon: float"})
//     .SetKernelFn(PD_KERNEL(RMSLnFwd))
//     .SetInferShapeFn(PD_INFER_SHAPE(RMSLnFwdInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(RMSLnFwdInferDtype));



// https://github.com/NVIDIA/apex/blob/85e9eddece9d4ac72b48c2407f8162f2173e1bf4/csrc/layer_norm_cuda_kernel.cu#L679
