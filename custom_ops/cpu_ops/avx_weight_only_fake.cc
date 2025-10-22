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
#include "paddle/extension.h"
#include "paddle/phi/core/kernel_registry.h"

std::vector<paddle::Tensor> InvokeAvxWeightOnly(const paddle::Tensor &x,
                                                const paddle::Tensor &weight,
                                                const paddle::Tensor &w_bias,
                                                const std::string &alog,
                                                bool trans) {
  auto out_shape = x.shape();
  out_shape[out_shape.size() - 1] = weight.shape()[1];
  auto out = paddle::empty(out_shape, x.dtype(), paddle::CPUPlace());
  return {out};
}

std::vector<std::vector<int64_t>> AvxWeightOnlyInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> weigh_shape,
    std::vector<int64_t> weigh_bias_shape) {
  int m = 1;
  for (int i = 0; i < x_shape.size() - 1; i++) {
    m = m * x_shape[i];
  }
  return {std::vector<int64_t>{m, weigh_shape[1]}};
}

std::vector<paddle::DataType> AvxWeightOnlyInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType weight_dtype,
    paddle::DataType weight_bias_dtype) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(avx_weight_only)
    .Inputs({"x", "weight", "w_bias"})
    .Outputs({"out"})
    .Attrs({"alog: std::string", "trans:bool"})
    .SetKernelFn(PD_KERNEL(InvokeAvxWeightOnly))
    .SetInferShapeFn(PD_INFER_SHAPE(AvxWeightOnlyInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AvxWeightOnlyInferDtype));
