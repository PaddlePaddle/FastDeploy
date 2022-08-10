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

#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/function/reduce.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {

#ifdef ENABLE_FDTENSOR_FUNC
TEST(fastdeploy, reduce_max) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int> expected_result_axis0 = {7, 4, 5};
  std::vector<int> expected_result_axis1 = {4, 7};
  std::vector<int> expected_result_noaxis = {7};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Max(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Max(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Max(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Max(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_max_large_dim) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5, 6, 9};
  std::vector<int> expected_result_axis0 = {4, 7, 5, 9};
  input.SetExternalData({2, 1, 2, 1, 2}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Max(input, &output, {4}, true);
  check_shape(output.shape, {2, 1, 2, 1, 1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Max(input, &output, {4});
  check_shape(output.shape, {2, 1, 2, 1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());
}

TEST(fastdeploy, reduce_min) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int> expected_result_axis0 = {2, 1, 3};
  std::vector<int> expected_result_axis1 = {2, 1};
  std::vector<int> expected_result_noaxis = {1};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Min(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Min(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Min(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Min(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_sum) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int> expected_result_axis0 = {9, 5, 8};
  std::vector<int> expected_result_axis1 = {9, 13};
  std::vector<int> expected_result_noaxis = {22};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Sum(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Sum(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Sum(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Sum(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_prod) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int> expected_result_axis0 = {14, 4, 15};
  std::vector<int> expected_result_axis1 = {24, 35};
  std::vector<int> expected_result_noaxis = {840};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Prod(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Prod(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Prod(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Prod(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_mean) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int> expected_result_axis0 = {4, 2, 4};
  std::vector<int> expected_result_axis1 = {3, 4};
  std::vector<int> expected_result_noaxis = {3};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  // keep_dim = true, reduce_all = false
  Mean(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Mean(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Mean(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Mean(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const int*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_all) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::array<bool, 6> inputs = {false, false, true, true, false, true};
  std::array<bool, 3> expected_result_axis0 = {false, false, true};
  std::array<bool, 2> expected_result_axis1 = {false, false};
  std::array<bool, 1> expected_result_noaxis = {false};

  input.SetExternalData({2, 3}, FDDataType::BOOL, inputs.data());

  // keep_dim = true, reduce_all = false
  All(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  All(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  All(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  All(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}

TEST(fastdeploy, reduce_any) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;

  std::array<bool, 6> inputs = {false, false, true, true, false, true};
  std::array<bool, 3> expected_result_axis0 = {true, false, true};
  std::array<bool, 2> expected_result_axis1 = {true, true};
  std::array<bool, 1> expected_result_noaxis = {true};

  input.SetExternalData({2, 3}, FDDataType::BOOL, inputs.data());

  // keep_dim = true, reduce_all = false
  Any(input, &output, {0}, true);
  check_shape(output.shape, {1, 3});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // keep_dim = false, reduce_all = false
  Any(input, &output, {1});
  check_shape(output.shape, {2});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());

  // keep_dim = false, reduce_all = true
  Any(input, &output, {1}, false, true);
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());

  // test 1-D tensor
  input.shape = {6};
  Any(input, &output, {0});
  check_shape(output.shape, {1});
  check_data(reinterpret_cast<const bool*>(output.Data()),
             expected_result_noaxis.data(), expected_result_noaxis.size());
}
#endif
}  // namespace fastdeploy
