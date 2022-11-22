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
  // Shape: [2, 3, 4]
  std::vector<float> x_data = {
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  // Shape: [2, 3, 4]
  std::vector<float> y_data = {
      0.8345295,  0.551608,   0.77101785, 0.386742,   0.12658621, 0.41240612,
      0.20051356, 0.68455917, 0.37947154, 0.2953741,  0.97703844, 0.2931625,
      0.2344262,  0.5054064,  0.40617892, 0.16315177, 0.71458364, 0.3748885,
      0.65257984, 0.83870554, 0.55464447, 0.38836837, 0.472637,   0.5546991};
  return std::make_tuple(x_data, y_data);
}

std::tuple<std::vector<float>, std::vector<float>> CreateBroadcastDim1Data() {
  // Shape: [2, 3, 4]
  std::vector<float> x_data = {
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  // Shape: [2, 1, 1]
  std::vector<float> y_data = {0.97375137, 0.11732706};
  return std::make_tuple(x_data, y_data);
}

std::tuple<std::vector<float>, std::vector<float>> CreateBroadcastDim2Data() {
  // Shape: [2, 3, 4]
  std::vector<float> x_data = {
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  // Shape: [1, 3, 1]
  std::vector<float> y_data = {0.30803263, 0.41172066, 0.5588573};
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

TEST(fastdeploy, check_broadcast_dim1) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y, z;

  auto test_data = CreateBroadcastDim1Data();
  auto x_data = std::get<0>(test_data);
  auto y_data = std::get<1>(test_data);
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, x_data.data());
  y.SetExternalData({2, 1, 1}, FDDataType::FP32, y_data.data());

  // Test Add functions
  std::vector<float> add_result = {
      1.816614, 1.619943, 1.111156, 1.088058, 1.633677, 1.509567,
      1.716668, 1.819356, 1.186033, 1.273452, 1.835923, 1.382693,
      0.244174, 0.273947, 0.546168, 0.96498,  0.36319,  0.786373,
      0.99621,  0.793586, 0.78378,  0.442557, 0.531266, 0.951468};

  Add(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());

  // Test subtract
  std::vector<float> sub_result = {
      -0.130889, -0.32756,  -0.836347, -0.859444, -0.313825, -0.437935,
      -0.230835, -0.128146, -0.76147,  -0.674051, -0.11158,  -0.56481,
      0.00952,   0.039292,  0.311514,  0.730326,  0.128536,  0.551719,
      0.761556,  0.558932,  0.549126,  0.207903,  0.296612,  0.716814};
  Subtract(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());

  // Test multiply
  std::vector<float> mul_result = {
      0.820738, 0.62923,  0.133798, 0.111307, 0.642604, 0.521752,
      0.723416, 0.823409, 0.20671,  0.291834, 0.83954,  0.398207,
      0.014883, 0.018376, 0.050315, 0.099453, 0.028846, 0.078497,
      0.103117, 0.079343, 0.078193, 0.038158, 0.048566, 0.097867};
  Multiply(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());

  // Test divide
  std::vector<float> div_result = {
      0.865583, 0.66361,  0.141108, 0.117388, 0.677715, 0.55026,
      0.762942, 0.868399, 0.218004, 0.30778,  0.885412, 0.419965,
      1.081139, 1.334897, 3.65509,  7.224699, 2.095538, 5.702402,
      7.490881, 5.763879, 5.680301, 2.771997, 3.528076, 7.109533};
  Divide(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
}

TEST(fastdeploy, check_broadcast_dim2) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y, z;

  auto test_data = CreateBroadcastDim2Data();
  auto x_data = std::get<0>(test_data);
  auto y_data = std::get<1>(test_data);
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, x_data.data());
  y.SetExternalData({1, 3, 1}, FDDataType::FP32, y_data.data());

  // Test Add functions
  std::vector<float> add_result = {
      1.150895, 0.954224, 0.445437, 0.42234,  1.071647, 0.947537,
      1.154637, 1.257326, 0.771139, 0.858558, 1.421029, 0.967799,
      0.43488,  0.464652, 0.736874, 1.155685, 0.657584, 1.080767,
      1.290604, 1.08798,  1.22531,  0.884088, 0.972796, 1.392998};

  Add(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());

  // Test subtract
  std::vector<float> sub_result = {
      0.53483,   0.338159,  -0.170628, -0.193726, 0.248205,  0.124095,
      0.331196,  0.433884,  -0.346576, -0.259156, 0.303314,  -0.149916,
      -0.181186, -0.151413, 0.120808,  0.53962,   -0.165857, 0.257325,
      0.467162,  0.264538,  0.107596,  -0.233627, -0.144919, 0.275283};
  Subtract(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());

  // Test multiply
  std::vector<float> mul_result = {
      0.259629, 0.199048, 0.042325, 0.03521,  0.271705, 0.220607,
      0.305874, 0.348153, 0.118635, 0.16749,  0.481831, 0.22854,
      0.039073, 0.048244, 0.132097, 0.261105, 0.101227, 0.27546,
      0.361854, 0.27843,  0.372452, 0.181757, 0.231333, 0.466166};
  Multiply(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());

  // Test divide
  std::vector<float> div_result = {
      2.736277, 2.097801, 0.446071, 0.371087, 1.602849, 1.301407,
      1.804418, 2.053832, 0.37985,  0.536274, 1.54274,  0.731745,
      0.411797, 0.508451, 1.392193, 2.751827, 0.59716,  1.625,
      2.134659, 1.642519, 1.192528, 0.581956, 0.740688, 1.492582};
  Divide(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
}

}  // namespace function
}  // namespace fastdeploy