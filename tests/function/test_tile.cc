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
#include "fastdeploy/function/tile.h"
#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"
#include <vector>

namespace fastdeploy {
namespace function {

std::vector<float> CreateTestData() {
  // Shape: [2, 3]
  std::vector<float> x_data = {0.8428625,  0.6461913, 0.13740455,
                               0.11430702, 0.659926,  0.535816};
  return x_data;
}

TEST(fastdeploy, tile) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3}, FDDataType::FP32, test_data.data());
  Tile(x, {2}, &y);
  std::vector<float> result = {0.842862, 0.646191, 0.137405, 0.842862,
                               0.646191, 0.137405, 0.114307, 0.659926,
                               0.535816, 0.114307, 0.659926, 0.535816};
  check_shape(y.Shape(), {2, 6});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {0.842862, 0.646191, 0.137405, 0.842862, 0.646191, 0.137405,
            0.842862, 0.646191, 0.137405, 0.114307, 0.659926, 0.535816,
            0.114307, 0.659926, 0.535816, 0.114307, 0.659926, 0.535816,
            0.842862, 0.646191, 0.137405, 0.842862, 0.646191, 0.137405,
            0.842862, 0.646191, 0.137405, 0.114307, 0.659926, 0.535816,
            0.114307, 0.659926, 0.535816, 0.114307, 0.659926, 0.535816};
  Tile(x, {2, 3}, &y);
  check_shape(y.Shape(), {4, 9});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {0.842862, 0.646191, 0.137405, 0.842862, 0.646191, 0.137405,
            0.114307, 0.659926, 0.535816, 0.114307, 0.659926, 0.535816,
            0.842862, 0.646191, 0.137405, 0.842862, 0.646191, 0.137405,
            0.114307, 0.659926, 0.535816, 0.114307, 0.659926, 0.535816,
            0.842862, 0.646191, 0.137405, 0.842862, 0.646191, 0.137405,
            0.114307, 0.659926, 0.535816, 0.114307, 0.659926, 0.535816};
  Tile(x, {3, 2}, &y);
  check_shape(y.Shape(), {6, 6});
}

}  // namespace function
}  // namespace fastdeploy
