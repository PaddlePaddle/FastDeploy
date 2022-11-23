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
#include "fastdeploy/function/clip.h"
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

TEST(fastdeploy, clip) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Clip(x, /* min = */ 0.2, /* max = */ 0.8, &y);
  std::vector<float> result = {
      0.8,      0.646191, 0.2, 0.2,      0.659926, 0.535816, 0.742916, 0.8,
      0.212282, 0.299701, 0.8, 0.408941, 0.2,      0.2,      0.428841, 0.8,
      0.245863, 0.669046, 0.8, 0.676259, 0.666453, 0.32523,  0.413939, 0.8};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy
