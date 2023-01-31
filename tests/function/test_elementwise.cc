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
#include "fastdeploy/function/elementwise.h"
#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"
#include <array>
#include <tuple>
#include <vector>

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

std::tuple<std::vector<float>, std::vector<float>> CreateBroadcastDim3Data() {
  // Shape: [2, 3, 4]
  std::vector<float> x_data = {
      0.8428625,  0.6461913, 0.13740455, 0.11430702, 0.659926,  0.535816,
      0.7429162,  0.8456049, 0.21228176, 0.29970083, 0.8621713, 0.40894133,
      0.12684688, 0.1566195, 0.42884097, 0.8476526,  0.2458633, 0.669046,
      0.87888306, 0.6762589, 0.666453,   0.32523027, 0.4139388, 0.8341406};
  // Shape: [1, 1, 4]
  std::vector<float> y_data = {0.62653106, 0.5128424, 0.9891219, 0.32416528};
  return std::make_tuple(x_data, y_data);
}

std::tuple<std::vector<float>, std::vector<float>> CreateBroadcastDim4Data() {
  // Shape: [2, 1, 4]
  std::vector<float> x_data = {0.8428625, 0.6461913, 0.13740455, 0.11430702,
                               0.659926,  0.535816,  0.7429162,  0.8456049};
  // Shape: [2, 2, 1]
  std::vector<float> y_data = {0.62653106, 0.5128424, 0.9891219, 0.32416528};
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
  z = x + y;
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
  z = x - y;
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
  z = x * y;
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
  z = x / y;
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());

  // Test Maximum
  std::vector<float> maximum_result = {
      0.842862, 0.646191, 0.771018, 0.386742, 0.659926, 0.535816,
      0.742916, 0.845605, 0.379472, 0.299701, 0.977038, 0.408941,
      0.234426, 0.505406, 0.428841, 0.847653, 0.714584, 0.669046,
      0.878883, 0.838706, 0.666453, 0.388368, 0.472637, 0.834141};
  Maximum(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), maximum_result.data(),
             maximum_result.size());

  x = 1.0f - x;
  sub_result = {0.157138, 0.353809, 0.862595, 0.885693, 0.340074, 0.464184,
                0.257084, 0.154395, 0.787718, 0.700299, 0.137829, 0.591059,
                0.873153, 0.843381, 0.571159, 0.152347, 0.754137, 0.330954,
                0.121117, 0.323741, 0.333547, 0.67477,  0.586061, 0.165859};
  check_shape(x.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(x.Data()), sub_result.data(),
             sub_result.size());
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
  z = x + y;
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
  z = x - y;
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
  z = x * y;
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
  z = x / y;
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());

  // Test Maximum
  std::vector<float> maximum_result = {
      0.973751, 0.973751, 0.973751, 0.973751, 0.973751, 0.973751,
      0.973751, 0.973751, 0.973751, 0.973751, 0.973751, 0.973751,
      0.126847, 0.15662,  0.428841, 0.847653, 0.245863, 0.669046,
      0.878883, 0.676259, 0.666453, 0.32523,  0.413939, 0.834141};
  Maximum(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), maximum_result.data(),
             maximum_result.size());
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
  z = x + y;
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
  z = x - y;
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

  // Test Maximum
  std::vector<float> maximum_result = {
      0.842862, 0.646191, 0.308033, 0.308033, 0.659926, 0.535816,
      0.742916, 0.845605, 0.558857, 0.558857, 0.862171, 0.558857,
      0.308033, 0.308033, 0.428841, 0.847653, 0.411721, 0.669046,
      0.878883, 0.676259, 0.666453, 0.558857, 0.558857, 0.834141};
  Maximum(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), maximum_result.data(),
             maximum_result.size());
}

TEST(fastdeploy, check_broadcast_dim3) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y, z;

  auto test_data = CreateBroadcastDim3Data();
  auto x_data = std::get<0>(test_data);
  auto y_data = std::get<1>(test_data);
  x.SetExternalData({2, 3, 4}, FDDataType::FP32, x_data.data());
  y.SetExternalData({4}, FDDataType::FP32, y_data.data());

  // Test Add functions
  std::vector<float> add_result = {
      1.469393, 1.159034, 1.126526, 0.438472, 1.286457, 1.048658,
      1.732038, 1.16977,  0.838813, 0.812543, 1.851293, 0.733107,
      0.753378, 0.669462, 1.417963, 1.171818, 0.872394, 1.181888,
      1.868005, 1.000424, 1.292984, 0.838073, 1.403061, 1.158306};

  Add(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());
  z = x + y;
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());

  // Test subtract
  std::vector<float> sub_result = {
      0.216331,  0.133349,  -0.851717, -0.209858, 0.033395,  0.022974,
      -0.246206, 0.52144,   -0.414249, -0.213142, -0.126951, 0.084776,
      -0.499684, -0.356223, -0.560281, 0.523487,  -0.380668, 0.156204,
      -0.110239, 0.352094,  0.039922,  -0.187612, -0.575183, 0.509975};
  Subtract(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());
  z = x - y;
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());
  // Test multiply
  std::vector<float> mul_result = {
      0.52808,  0.331394, 0.13591,  0.037054, 0.413464, 0.274789,
      0.734835, 0.274116, 0.133001, 0.153699, 0.852793, 0.132565,
      0.079474, 0.080321, 0.424176, 0.27478,  0.154041, 0.343115,
      0.869322, 0.21922,  0.417554, 0.166792, 0.409436, 0.270399};
  Multiply(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());
  z = x * y;
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());
  // Test divide
  std::vector<float> div_result = {
      1.345284, 1.260019, 0.138916, 0.35262,  1.053301, 1.044797,
      0.751087, 2.608561, 0.338821, 0.584392, 0.871653, 1.261521,
      0.202459, 0.305395, 0.433557, 2.614878, 0.39242,  1.304584,
      0.888549, 2.086155, 1.063719, 0.634172, 0.418491, 2.573195};
  Divide(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
  z = x / y;
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());

  // Test Maximum
  std::vector<float> maximum_result = {
      0.842862, 0.646191, 0.989122, 0.324165, 0.659926, 0.535816,
      0.989122, 0.845605, 0.626531, 0.512842, 0.989122, 0.408941,
      0.626531, 0.512842, 0.989122, 0.847653, 0.626531, 0.669046,
      0.989122, 0.676259, 0.666453, 0.512842, 0.989122, 0.834141};
  Maximum(x, y, &z);
  check_shape(z.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), maximum_result.data(),
             maximum_result.size());
}

TEST(fastdeploy, check_broadcast_dim4) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y, z;

  auto test_data = CreateBroadcastDim4Data();
  auto x_data = std::get<0>(test_data);
  auto y_data = std::get<1>(test_data);
  x.SetExternalData({2, 1, 4}, FDDataType::FP32, x_data.data());
  y.SetExternalData({2, 2, 1}, FDDataType::FP32, y_data.data());

  // Test Add functions
  std::vector<float> add_result = {1.469393, 1.272722, 0.763936, 0.740838,
                                   1.355705, 1.159034, 0.650247, 0.627149,
                                   1.649048, 1.524938, 1.732038, 1.834727,
                                   0.984091, 0.859981, 1.067081, 1.16977};

  Add(x, y, &z);
  check_shape(z.shape, {2, 2, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());

  z = x + y;
  check_data(reinterpret_cast<const float*>(z.Data()), add_result.data(),
             add_result.size());
  // Test subtract
  std::vector<float> sub_result = {0.216331,  0.01966,   -0.489127, -0.512224,
                                   0.33002,   0.133349,  -0.375438, -0.398535,
                                   -0.329196, -0.453306, -0.246206, -0.143517,
                                   0.335761,  0.211651,  0.418751,  0.52144};
  Subtract(x, y, &z);
  check_shape(z.shape, {2, 2, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());
  z = x - y;
  check_data(reinterpret_cast<const float*>(z.Data()), sub_result.data(),
             sub_result.size());
  // Test multiply
  std::vector<float> mul_result = {0.52808,  0.404859, 0.086088, 0.071617,
                                   0.432256, 0.331394, 0.070467, 0.058621,
                                   0.652747, 0.529987, 0.734835, 0.836406,
                                   0.213925, 0.173693, 0.240828, 0.274116};
  Multiply(x, y, &z);
  check_shape(z.shape, {2, 2, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());
  z = x * y;
  check_data(reinterpret_cast<const float*>(z.Data()), mul_result.data(),
             mul_result.size());

  // Test divide
  std::vector<float> div_result = {1.345284, 1.031379, 0.21931,  0.182444,
                                   1.643512, 1.260019, 0.267927, 0.222889,
                                   0.667184, 0.541709, 0.751087, 0.854905,
                                   2.03577,  1.65291,  2.291782, 2.608561};
  Divide(x, y, &z);
  check_shape(z.shape, {2, 2, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
  z = x / y;
  check_data(reinterpret_cast<const float*>(z.Data()), div_result.data(),
             div_result.size());
  // Test Maximum
  std::vector<float> maximum_result = {0.842862, 0.646191, 0.626531, 0.626531,
                                       0.842862, 0.646191, 0.512842, 0.512842,
                                       0.989122, 0.989122, 0.989122, 0.989122,
                                       0.659926, 0.535816, 0.742916, 0.845605};
  Maximum(x, y, &z);
  check_shape(z.shape, {2, 2, 4});
  check_data(reinterpret_cast<const float*>(z.Data()), maximum_result.data(),
             maximum_result.size());
}

TEST(fastdeploy, mixed_operation) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor a, b, c, d, e, output;

  auto test_data = CreateSameDimeData();
  auto a_data = std::get<0>(test_data);
  auto b_data = std::get<1>(test_data);
  auto c_data = std::get<1>(CreateBroadcastDim1Data());
  auto d_data = std::get<1>(CreateBroadcastDim2Data());
  auto e_data = std::get<1>(CreateBroadcastDim3Data());

  a.SetExternalData({2, 3, 4}, FDDataType::FP32, a_data.data());
  b.SetExternalData({2, 3, 4}, FDDataType::FP32, b_data.data());
  c.SetExternalData({2, 1, 1}, FDDataType::FP32, c_data.data());
  d.SetExternalData({1, 3, 1}, FDDataType::FP32, d_data.data());
  e.SetExternalData({1, 1, 4}, FDDataType::FP32, e_data.data());

  std::vector<float> result = {
      3.238058,  3.004797,  2.278015,  2.881238,  1.822084,  2.073209,
      1.524921,  2.619779,  1.196421,  1.318079,  1.59565,   1.538118,
      -0.215903, -0.052794, -0.434044, 0.195022,  -0.165874, 0.022943,
      -0.130613, 0.527984,  -0.046946, -0.176592, -0.583538, 0.348473};

  output = a * b + c / d - e;
  check_shape(output.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(output.Data()), result.data(),
             result.size());

  result = {2.854443,  1.87709,   1.585621,  1.012709,  0.332781,  0.998346,
            0.228024,  2.140475,  0.246941,  0.301517,  1.575438,  0.595582,
            -0.410393, -0.163718, -0.405571, 0.58563,   -0.177035, 0.263035,
            0.075725,  0.591098,  0.156365,  -0.106078, -0.475957, 0.626429};
  output = a + b * c / d - e;
  check_shape(output.shape, {2, 3, 4});
  check_data(reinterpret_cast<const float*>(output.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy