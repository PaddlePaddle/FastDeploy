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
#include "fastdeploy/function/isfinite.h"
#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"
#include <array>
#include <cmath>
#include <vector>

namespace fastdeploy {
namespace function {

std::vector<float> CreateTestData() {
  // Shape: [2, 3]
  std::vector<float> x_data = {0.8428625,  NAN,      INFINITY,
                               0.11430702, 0.659926, 0.535816};
  return x_data;
}

TEST(fastdeploy, finite) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3}, FDDataType::FP32, test_data.data());

  std::array<bool, 6> result = {false, true, false, false, false, false};
  IsNan(x, &y);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const bool*>(y.Data()), result.data(),
             result.size());

  std::vector<int> int_result = {0, 1, 0, 0, 0, 0};
  IsNan(x, &y, FDDataType::INT32);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const int*>(y.Data()), int_result.data(),
             int_result.size());

  result = {false, false, true, false, false, false};
  IsInf(x, &y);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const bool*>(y.Data()), result.data(),
             result.size());

  int_result = {0, 0, 1, 0, 0, 0};
  IsInf(x, &y, FDDataType::INT32);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const int*>(y.Data()), int_result.data(),
             int_result.size());

  result = {true, false, false, true, true, true};
  IsFinite(x, &y);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const bool*>(y.Data()), result.data(),
             result.size());

  int_result = {1, 0, 0, 1, 1, 1};
  IsFinite(x, &y, FDDataType::INT32);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const int*>(y.Data()), int_result.data(),
             int_result.size());
}

}  // namespace function
}  // namespace fastdeploy