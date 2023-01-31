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
#include "fastdeploy/function/sort.h"
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

TEST(fastdeploy, sort_dim0) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, out, indices;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Sort(x, &out, &indices, 0);

  std::vector<float> out_result = {
      0.126847, 0.15662,  0.137405, 0.114307, 0.245863, 0.535816,
      0.742916, 0.676259, 0.212282, 0.299701, 0.413939, 0.408941,
      0.842862, 0.646191, 0.428841, 0.847653, 0.659926, 0.669046,
      0.878883, 0.845605, 0.666453, 0.32523,  0.862171, 0.834141};
  std::vector<int64_t> indices_result = {1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                                         0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
  check_shape(out.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(out.Data()), out_result.data(),
             out_result.size());
  check_shape(indices.shape, {2, 3, 4});
  check_data(reinterpret_cast<const int64_t*>(indices.Data()),
             indices_result.data(), indices_result.size());
}

TEST(fastdeploy, sort_dim1) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, out, indices;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Sort(x, &out, &indices, 1);

  std::vector<float> out_result = {
      0.212282, 0.299701, 0.137405, 0.114307, 0.659926, 0.535816,
      0.742916, 0.408941, 0.842862, 0.646191, 0.862171, 0.845605,
      0.126847, 0.15662,  0.413939, 0.676259, 0.245863, 0.32523,
      0.428841, 0.834141, 0.666453, 0.669046, 0.878883, 0.847653};
  std::vector<int64_t> indices_result = {2, 2, 0, 0, 1, 1, 1, 2, 0, 0, 2, 1,
                                         0, 0, 2, 1, 1, 2, 0, 2, 2, 1, 1, 0};
  check_shape(out.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(out.Data()), out_result.data(),
             out_result.size());
  check_shape(indices.shape, {2, 3, 4});
  check_data(reinterpret_cast<const int64_t*>(indices.Data()),
             indices_result.data(), indices_result.size());
}

TEST(fastdeploy, sort_dim2) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, out, indices;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  Sort(x, &out, &indices, 2);

  std::vector<float> out_result = {
      0.114307, 0.137405, 0.646191, 0.842862, 0.535816, 0.659926,
      0.742916, 0.845605, 0.212282, 0.299701, 0.408941, 0.862171,
      0.126847, 0.15662,  0.428841, 0.847653, 0.245863, 0.669046,
      0.676259, 0.878883, 0.32523,  0.413939, 0.666453, 0.834141};
  std::vector<int64_t> indices_result = {3, 2, 1, 0, 1, 0, 2, 3, 0, 1, 3, 2,
                                         0, 1, 2, 3, 0, 1, 3, 2, 1, 2, 0, 3};
  check_shape(out.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(out.Data()), out_result.data(),
             out_result.size());
  check_shape(indices.shape, {2, 3, 4});
  check_data(reinterpret_cast<const int64_t*>(indices.Data()),
             indices_result.data(), indices_result.size());
}

}  // namespace function
}  // namespace fastdeploy