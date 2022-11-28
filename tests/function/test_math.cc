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
#include "fastdeploy/function/math.h"
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

TEST(fastdeploy, exp_sqrt_round_log) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  auto test_data = CreateTestData();
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, test_data.data());

  // Test Sqrt function
  Sqrt(x, &y);
  std::vector<float> sqrt_result = {
      0.918075, 0.80386,  0.370681, 0.338093, 0.812358, 0.731995,
      0.861926, 0.919568, 0.46074,  0.547449, 0.928532, 0.639485,
      0.356156, 0.395752, 0.654859, 0.920681, 0.495846, 0.817952,
      0.937488, 0.82235,  0.816366, 0.57029,  0.643381, 0.913313};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), sqrt_result.data(),
             sqrt_result.size());

  // Test Exp function
  Exp(x, &y);
  std::vector<float> exp_result = {
      2.323007, 1.908259, 1.147292, 1.121096, 1.934649, 1.708842,
      2.102057, 2.329386, 1.236496, 1.349455, 2.368297, 1.505223,
      1.135243, 1.169551, 1.535477, 2.334161, 1.278725, 1.952374,
      2.408208, 1.966507, 1.947318, 1.384349, 1.512764, 2.302834};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), exp_result.data(),
             exp_result.size());

  // Test Round function
  Round(x, &y);
  std::vector<float> round_result = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), round_result.data(),
             round_result.size());

  Ceil(x, &y);
  std::vector<float> ceil_result = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), ceil_result.data(),
             ceil_result.size());

  Floor(x, &y);
  std::vector<float> floor_result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), floor_result.data(),
             floor_result.size());

  // Test Log function
  Log(x, &y);
  std::vector<float> log_result = {
      -0.170951, -0.43666,  -1.984826, -2.168867, -0.415628, -0.623964,
      -0.297172, -0.167703, -1.549841, -1.204971, -0.148301, -0.894184,
      -2.064775, -1.853936, -0.846669, -0.165284, -1.40298,  -0.401902,
      -0.129103, -0.391179, -0.405786, -1.123222, -0.882037, -0.181353};
  check_shape(y.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(y.Data()), log_result.data(),
             log_result.size());
}

TEST(fastdeploy, abs) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  std::vector<float> test_data = {-1, 2, 3, -5, -4, -6};
  x.SetExternalData({2, 3}, FDDataType::FP32, test_data.data());
  std::vector<float> result = {1, 2, 3, 5, 4, 6};
  Abs(x, &y);
  check_shape(y.shape, {2, 3});
  check_data(reinterpret_cast<const float*>(y.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy