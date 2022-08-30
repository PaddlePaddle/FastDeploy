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
#include "fastdeploy/function/softmax.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {

TEST(fastdeploy, softmax) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;
  std::vector<float> inputs = {1, 2, 3, 4, 5, 6};
  std::vector<float> expected_result_axis0 = {
      0.04742587, 0.04742587, 0.04742587, 0.95257413, 0.95257413, 0.95257413};
  std::vector<float> expected_result_axis1 = {
      0.09003057, 0.24472846, 0.66524088, 0.09003057, 0.24472846, 0.66524088};
  input.SetExternalData({2, 3}, FDDataType::FP32, inputs.data());

  // axis = 0
  Softmax(input, &output, 0);
  check_shape(output.shape, {2, 3});
  check_data(reinterpret_cast<const float*>(output.Data()),
             expected_result_axis0.data(), expected_result_axis0.size());

  // axis = 1
  Softmax(input, &output, 1);
  check_shape(output.shape, {2, 3});
  check_data(reinterpret_cast<const float*>(output.Data()),
             expected_result_axis1.data(), expected_result_axis1.size());
}

}  // namespace fastdeploy