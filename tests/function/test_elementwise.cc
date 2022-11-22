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

#include <array>
#include <vector>
#include <tuple>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/function/elementwise.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {
namespace function {

std::tuple<std::vector<float>, std::vector<float>> CreateSameDimeData() {
  std::vector<float> x_data = {
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  std::vector<float> y_data = {
      0.8345295,  0.551608,   0.77101785, 0.386742,   0.12658621, 0.41240612,
      0.20051356, 0.68455917, 0.37947154, 0.2953741,  0.97703844, 0.2931625,
      0.2344262,  0.5054064,  0.40617892, 0.16315177, 0.71458364, 0.3748885,
      0.65257984, 0.83870554, 0.55464447, 0.38836837, 0.472637,   0.5546991};
  return std::make_tuple(x_data, y_data);
}

TEST(fastdeploy, check_same_dim) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y, z;

  auto test_data = CreateSameDimeData();
  auto x_data = std::get<0>(test_data);
  auto y_data = std::get<1>(test_data);
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, x_data.data());
  y.SetExternalData({2, 3, 4}, FDDataType::FP32, y_data.data());

  // Test Add functions
  std::vector<float> add_result = {
      1.677392,   1.1977993,  0.9084224, 0.50104904, 0.7865122,  0.94822216,
      0.94342977, 1.530164,   0.5917533, 0.5950749,  1.8392098,  0.70210385,
      0.36127308, 0.66202587, 0.8350199, 1.0108044,  0.96044695, 1.0439345,
      1.5314629,  1.5149645,  1.2210975, 0.7135986,  0.8865758,  1.3888397};

  Add(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());

  // Test subtract
  std::vector<float> sub_result = {
      0.008332968, 0.09458327,  -0.6336133,   -0.27243498, 0.5333398,
      0.1234099,   0.5424027,   0.16104573,   -0.16718978, 0.004326731,
      -0.11486715, 0.11577883,  -0.10757932,  -0.3487869,  0.022662044,
      0.6845008,   -0.46872032, 0.29415748,   0.22630322,  -0.16244662,
      0.11180854,  -0.0631381,  -0.058698207, 0.27944148};
  Subtract(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());

  // Test multiply
  std::vector<float> mul_result = {
      0.70339364, 0.3564443,  0.105941355, 0.044207327, 0.083537534,
      0.2209738,  0.14896478, 0.5788666,   0.08055489,  0.08852386,
      0.8423745,  0.11988626, 0.029736232, 0.079156496, 0.17418616,
      0.13829602, 0.17568989, 0.25081766,  0.57354134,  0.5671821,
      0.36964446, 0.12630916, 0.19564278,  0.46269706};
  Multiply(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());

  // Test divide
  std::vector<float> div_result = {
      1.0099852,  1.1714683,  0.17821188, 0.29556403, 5.2132535,  1.2992436,
      3.7050674,  1.2352546,  0.5594142,  1.0146483,  0.88243335, 1.3949306,
      0.54109514, 0.30988827, 1.0557933,  5.195485,   0.34406513, 1.7846532,
      1.3467824,  0.8063127,  1.201586,   0.8374273,  0.875807,   1.5037713};
  Divide(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
}

}  // namespace function
}  // namespace fastdeploy