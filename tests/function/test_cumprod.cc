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
#include "fastdeploy/function/cumprod.h"
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
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  return x_data;
}

TEST(fastdeploy, cumprod) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  std::vector<float> result = {0.842862, 0.646191, 0.137405, 0.114307, 0.659926,
                               0.535816, 0.742916, 0.845605, 0.212282, 0.299701,
                               0.862171, 0.408941, 0.106914, 0.101206, 0.058925,
                               0.096893, 0.162252, 0.358486, 0.652937, 0.571848,
                               0.141476, 0.097472, 0.356886, 0.341115};
  Cumprod(x, &y, 0);
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {0.842862, 0.646191, 0.137405, 0.114307, 0.556227, 0.34624,
            0.10208,  0.096659, 0.118077, 0.103768, 0.088011, 0.039528,
            0.126847, 0.15662,  0.428841, 0.847653, 0.031187, 0.104786,
            0.376901, 0.573233, 0.020785, 0.034079, 0.156014, 0.478157};
  Cumprod(x, &y, 1);
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());

  result = {0.842862, 0.54465,  0.074837, 0.008554, 0.659926, 0.353599,
            0.262694, 0.222136, 0.212282, 0.063621, 0.054852, 0.022431,
            0.126847, 0.019867, 0.00852,  0.007222, 0.245863, 0.164494,
            0.144571, 0.097767, 0.666453, 0.216751, 0.089722, 0.07484};
  Cumprod(x, &y, 2);
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy
