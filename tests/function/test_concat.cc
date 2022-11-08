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
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/function/concat.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {

TEST(fastdeploy, concat1) {
  CheckShape check_shape;
  std::vector<FDTensor> inputs(3);
  FDTensor output;
  inputs[0].Allocate({5, 1, 4, 5}, FDDataType::FP32);
  inputs[1].Allocate({5, 2, 4, 5}, FDDataType::FP32);
  inputs[2].Allocate({5, 3, 4, 5}, FDDataType::FP32);
  Concat(inputs, &output, 1);
  check_shape(output.shape, {5, 6, 4, 5});
}

TEST(fastdeploy, concat2) {
  CheckShape check_shape;
  std::vector<FDTensor> inputs(3);
  FDTensor output;
  inputs[0].Allocate({2, 3, 4, 5}, FDDataType::FP32);
  inputs[1].Allocate({2, 3, 4, 5}, FDDataType::FP32);
  inputs[2].Allocate({2, 3, 4, 5}, FDDataType::FP32);
  Concat(inputs, &output, 1);
  check_shape(output.shape, {2, 9, 4, 5});
}

TEST(fastdeploy, concat3) {
  CheckShape check_shape;
  std::vector<FDTensor> inputs(3);
  FDTensor output;
  inputs[0].Allocate({1, 256, 170, 256}, FDDataType::FP32);
  inputs[1].Allocate({1, 128, 170, 256}, FDDataType::FP32);
  inputs[2].Allocate({1, 128, 170, 256}, FDDataType::FP32);
  Concat(inputs, &output, 1);
  check_shape(output.shape, {1, 512, 170, 256});
}

TEST(fastdeploy, concat4) {
  CheckShape check_shape;
  std::vector<FDTensor> inputs(3);
  FDTensor output;
  inputs[0].Allocate({2, 3, 4, 5}, FDDataType::FP32);
  inputs[1].Allocate({2, 3, 4, 5}, FDDataType::FP32);
  inputs[2].Allocate({0, 3, 4, 5}, FDDataType::FP32);
  Concat(inputs, &output, 0);
  check_shape(output.shape, {4, 3, 4, 5});
}

TEST(fastdeploy, concat5) {
  CheckShape check_shape;
  std::vector<FDTensor> inputs(3);
  FDTensor output;
  inputs[0].Allocate({5, 1, 4, 5}, FDDataType::FP32);
  inputs[1].Allocate({5, 2, 4, 5}, FDDataType::FP32);
  inputs[2].Allocate({5, 3, 4, 5}, FDDataType::FP32);
  Concat(inputs, &output, -3);
  check_shape(output.shape, {5, 6, 4, 5});
}

}  // namespace fastdeploy
