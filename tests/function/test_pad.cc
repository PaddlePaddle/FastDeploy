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
#include "fastdeploy/function/pad.h"
#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"

namespace fastdeploy {
namespace function {
TEST(fastdeploy, pad_2d) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  std::vector<float> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<float> expected_result = {2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2,
                                        4,   3,   2.2, 2.2, 7,   1,   5,
                                        2.2, 2.2, 2.2, 2.2, 2.2, 2.2};
  input.SetExternalData({2, 3}, FDDataType::FP32, inputs.data());

  Pad(input, &output, {1, 1, 1, 1}, 2.2);
  check_shape(output.shape, {4, 5});
  check_data(reinterpret_cast<const float*>(output.Data()),
             expected_result.data(), expected_result.size());
  check_type(input.dtype, output.dtype);
}

TEST(fastdeploy, pad_2d_int32_t) {
  FDTensor input, output;
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  std::vector<int32_t> inputs = {2, 4, 3, 7, 1, 5};
  std::vector<int32_t> expected_result = {2, 2, 2, 2, 2, 2, 2, 4, 3, 2,
                                          2, 7, 1, 5, 2, 2, 2, 2, 2, 2};
  input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  Pad(input, &output, {1, 1, 1, 1}, 2.2);
  check_shape(output.shape, {4, 5});
  check_data(reinterpret_cast<const int32_t*>(output.Data()),
             expected_result.data(), expected_result.size());
  check_type(input.dtype, output.dtype);
}

}  // namespace function
}  // namespace fastdeploy
