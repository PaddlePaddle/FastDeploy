// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/function/elementwise.h"
#include "fastdeploy/function/eigen.h"
#include "fastdeploy/function/elementwise_base.h"
#include "fastdeploy/function/elementwise_functor.h"
#include "fastdeploy/utils/utils.h"
#include <algorithm>

namespace fastdeploy {
namespace function {

DEFINE_ELEMENTWISE_OP(Add);
DEFINE_ELEMENTWISE_OP(Multiply);
DEFINE_ELEMENTWISE_OP(Subtract);
DEFINE_ELEMENTWISE_OP(Divide);

void Add(const FDTensor& x, const FDTensor& y, FDTensor* out) {
  FD_VISIT_ALL_TYPES(x.dtype, "AddRawKernel",
                     ([&] { AddRawKernel<data_t>()(x, y, -1, out); }));
}

FDTensor operator+(const FDTensor& x, const FDTensor& y) {
  FDTensor out;
  Add(x, y, &out);
  return out;
}

void Subtract(const FDTensor& x, const FDTensor& y, FDTensor* out) {
  FD_VISIT_ALL_TYPES(x.dtype, "SubtractRawKernel",
                     ([&] { SubtractRawKernel<data_t>()(x, y, -1, out); }));
}

FDTensor operator-(const FDTensor& x, const FDTensor& y) {
  FDTensor out;
  Subtract(x, y, &out);
  return out;
}

void Multiply(const FDTensor& x, const FDTensor& y, FDTensor* out) {
  FD_VISIT_ALL_TYPES(x.dtype, "MultiplyRawKernel",
                     ([&] { MultiplyRawKernel<data_t>()(x, y, -1, out); }));
}

FDTensor operator*(const FDTensor& x, const FDTensor& y) {
  FDTensor out;
  Multiply(x, y, &out);
  return out;
}

void Divide(const FDTensor& x, const FDTensor& y, FDTensor* out) {
  FD_VISIT_ALL_TYPES(x.dtype, "DivideRawKernel",
                     ([&] { DivideRawKernel<data_t>()(x, y, -1, out); }));
}

FDTensor operator/(const FDTensor& x, const FDTensor& y) {
  FDTensor out;
  Divide(x, y, &out);
  return out;
}

}  // namespace function
}  // namespace fastdeploy
