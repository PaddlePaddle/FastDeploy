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

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/function/quantile.h"
#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"
#include <array>
#include <vector>

namespace fastdeploy {
namespace function {

std::vector<float> CreateTestData() {
  // Shape: [2, 3, 4]
  std::vector<float> x_data = {
      1.8428625,  0.6461913, 0.13740455,  0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, -1.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 2.1566195, -9.42884097, 20.8476526, 0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,    0.32523027, 0.4139388, 0.8341406};
  return x_data;
}

TEST(fastdeploy, quantile) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  std::vector<float> result = {1.834282, 2.149067, 0.089573, 20.743986,
                               0.657856, 0.66838,  0.878203, 0.844758,
                               0.657059, 0.325103, 0.85993,  0.832015};
  Quantile(x, {0.995}, {0}, &y);
  check_shape(y.shape, {3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {1.831033, 0.645088, 0.860979, 0.841238,
            0.662247, 2.141744, 0.874234, 20.647517};
  Quantile(x, {0.995}, {1}, &y);
  check_shape(y.shape, {2, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {1.824912, 0.844065, 0.855373, 20.567287, 0.875844, 0.831625};
  Quantile(x, {0.995}, {2}, &y);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy