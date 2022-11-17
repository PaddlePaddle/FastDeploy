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
#include <cstring>
#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {

TEST(fastdeploy, fd_tensor_constructor) {
  CheckShape check_shape;
  CheckData check_data;

  FDTensor tensor1;
  check_shape(tensor1.shape, {0});
  ASSERT_EQ(tensor1.name, "");
  ASSERT_EQ(tensor1.dtype, FDDataType::INT8);
  ASSERT_EQ(tensor1.device, Device::CPU);

  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  tensor1.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());
  ASSERT_EQ(tensor1.dtype, FDDataType::INT32);

  FDTensor tensor2(tensor1);
  check_shape(tensor1.shape, {2, 3});
  ASSERT_EQ(tensor2.name, "");
  ASSERT_EQ(tensor2.dtype, FDDataType::INT32);
  ASSERT_EQ(tensor2.device, Device::CPU);

  FDTensor tensor3;
  tensor3.Resize({2, 3}, FDDataType::INT32, "tensor3");
  check_shape(tensor3.shape, {2, 3});
  ASSERT_EQ(tensor3.Nbytes(), 24);

  // Copy constructor
  FDTensor tensor4(tensor3);
  check_shape(tensor4.shape, {2, 3});
  ASSERT_EQ(tensor3.Nbytes(), tensor4.Nbytes());
  check_data(reinterpret_cast<int*>(tensor3.Data()),
             reinterpret_cast<int*>(tensor4.Data()), tensor4.Numel());

  // Move constructor
  ASSERT_NE(tensor1.external_data_ptr, nullptr);
  FDTensor tensor5(std::move(tensor1));
  ASSERT_EQ(tensor1.external_data_ptr, nullptr);
  ASSERT_EQ(tensor5.external_data_ptr, inputs.data());
  check_shape(tensor5.shape, {2, 3});
}

TEST(fastdeploy, fd_tensor_assignment) {
  CheckShape check_shape;
  CheckData check_data;

  FDTensor tensor1("T1");
  std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
  tensor1.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

  FDTensor tensor2;
  tensor2 = tensor1;
  ASSERT_EQ(tensor2.name, "T1");
  ASSERT_EQ(tensor2.dtype, FDDataType::INT32);
  ASSERT_EQ(tensor2.device, Device::CPU);
  ASSERT_EQ(tensor2.Data(), inputs.data());
  check_shape(tensor2.shape, {2, 3});

  FDTensor tensor3;
  tensor3 = std::move(tensor1);
  ASSERT_EQ(tensor3.name, "T1");
  ASSERT_EQ(tensor3.dtype, FDDataType::INT32);
  ASSERT_EQ(tensor3.device, Device::CPU);
  ASSERT_EQ(tensor3.Data(), inputs.data());
  ASSERT_EQ(tensor1.Data(), nullptr);
}

}  // namespace fastdeploy