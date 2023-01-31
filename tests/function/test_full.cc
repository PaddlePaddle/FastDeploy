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
#include "fastdeploy/function/full.h"
#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"
#include <array>
#include <vector>

namespace fastdeploy {
namespace function {

TEST(fastdeploy, full) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor y;
  Full(1, {2, 3, 4}, &y);
  std::vector<float> result(24, 1);
  check_shape(y.Shape(), {2, 3, 4});
  check_data(reinterpret_cast<float*>(y.Data()),
             reinterpret_cast<float*>(result.data()), result.size());
}

TEST(fastdeploy, full_like) {
  CheckShape check_shape;
  CheckData check_data;
  FDTensor x, y;
  x.Allocate({3, 4}, FDDataType::FP32);
  FullLike(x, 0, &y, FDDataType::INT32);
  std::vector<int> result(12, 0);
  check_shape(y.Shape(), {3, 4});
  check_data(reinterpret_cast<int*>(y.Data()),
             reinterpret_cast<int*>(result.data()), result.size());
}

}  // namespace function
}  // namespace fastdeploy
