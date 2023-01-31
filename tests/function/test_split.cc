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
#include "fastdeploy/function/split.h"
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

TEST(fastdeploy, split_axis0) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x;
  std::vector<FDTensor> out;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Split(x, {1, 1}, &out, 0);
  ASSERT_EQ(out.size(), 2);
  check_shape(out[0].Shape(), {1, 3, 4});
  check_shape(out[1].Shape(), {1, 3, 4});
  std::vector<float> result1 = {0.842862, 0.646191, 0.137405, 0.114307,
                                0.659926, 0.535816, 0.742916, 0.845605,
                                0.212282, 0.299701, 0.862171, 0.408941};
  std::vector<float> result2 = {0.126847, 0.15662,  0.428841, 0.847653,
                                0.245863, 0.669046, 0.878883, 0.676259,
                                0.666453, 0.32523,  0.413939, 0.834141};
  check_data(reinterpret_cast<const float*>(out[0].Data()), result1.data(),
             result1.size());
  check_data(reinterpret_cast<const float*>(out[1].Data()), result2.data(),
             result2.size());
}

TEST(fastdeploy, split_axis1) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x;
  std::vector<FDTensor> out;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Split(x, {2, 1}, &out, 1);
  ASSERT_EQ(out.size(), 2);
  check_shape(out[0].Shape(), {2, 2, 4});
  check_shape(out[1].Shape(), {2, 1, 4});
  std::vector<float> result1 = {0.842862, 0.646191, 0.137405, 0.114307,
                                0.659926, 0.535816, 0.742916, 0.845605,
                                0.126847, 0.15662,  0.428841, 0.847653,
                                0.245863, 0.669046, 0.878883, 0.676259};
  std::vector<float> result2 = {0.212282, 0.299701, 0.862171, 0.408941,
                                0.666453, 0.32523,  0.413939, 0.834141};
  check_data(reinterpret_cast<const float*>(out[0].Data()), result1.data(),
             result1.size());
  check_data(reinterpret_cast<const float*>(out[1].Data()), result2.data(),
             result2.size());
}

TEST(fastdeploy, split_axis2) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x;
  std::vector<FDTensor> out;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Split(x, {1, 2, 1}, &out, 2);
  ASSERT_EQ(out.size(), 3);
  check_shape(out[0].Shape(), {2, 3, 1});
  check_shape(out[1].Shape(), {2, 3, 2});
  check_shape(out[2].Shape(), {2, 3, 1});
  std::vector<float> result1 = {0.842862, 0.659926, 0.212282,
                                0.126847, 0.245863, 0.666453};
  std::vector<float> result2 = {0.646191, 0.137405, 0.535816, 0.742916,
                                0.299701, 0.862171, 0.15662,  0.428841,
                                0.669046, 0.878883, 0.32523,  0.413939};
  std::vector<float> result3 = {0.114307, 0.845605, 0.408941,
                                0.847653, 0.676259, 0.834141};
  check_data(reinterpret_cast<const float*>(out[0].Data()), result1.data(),
             result1.size());
  check_data(reinterpret_cast<const float*>(out[1].Data()), result2.data(),
             result2.size());
  check_data(reinterpret_cast<const float*>(out[2].Data()), result3.data(),
             result3.size());
}

}  // namespace function
}  // namespace fastdeploy