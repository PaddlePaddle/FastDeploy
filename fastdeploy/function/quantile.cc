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

#include "fastdeploy/function/quantile.h"
#include "fastdeploy/function/transpose.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fastdeploy {
namespace function {

template <typename T>
void QuantileKernel(const FDTensor& x, const std::vector<double>& q,
                    const std::vector<int>& axis, FDTensor* out) {
  FDASSERT(q.size(), "q should not be empty.");
  FDASSERT(axis.size(), "axis should not be empty.");
  std::vector<int64_t> axis_src;
  std::vector<int64_t> out_shape = x.Shape();
  int64_t rank = x.Shape().size();
  for (auto axis_single : axis) {
    FDASSERT(axis_single >= -rank && axis_single < rank,
             "The axis is expected to be in range of [%d, %d), but got %d",
             -rank, rank, axis_single);
    if (axis_single < 0) {
      axis_single += rank;
    }
    axis_src.push_back(axis_single);
    out_shape[axis_single] = 1;
  }
  out->Allocate(out_shape, x.Dtype());
  std::vector<int64_t> axis_dst;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axis_src.begin(), axis_src.end(), i) == axis_src.end()) {
      axis_dst.push_back(i);
    }
  }
  axis_dst.insert(axis_dst.end(), axis_src.begin(), axis_src.end());
  FDTensor y;
  Transpose(x, &y, axis_dst);
  std::vector<int64_t> y_shape(rank - axis_src.size(), 0);
  y_shape.push_back(-1);
  y.Reshape({y_shape});
}

void Quantile(const FDTensor& x, const std::vector<double>& q,
              const std::vector<int>& axis, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "QuantileKernel",
                       ([&] { QuantileKernel<data_t>(x, q, axis, out); }));
}

}  // namespace function
}  // namespace fastdeploy