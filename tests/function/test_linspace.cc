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
#include "fastdeploy/function/linspace.h"
#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "gtest_utils.h"
#include "gtest/gtest.h"

namespace fastdeploy {
namespace function {

TEST(fastdeploy, linspace) {
  CheckShape check_shape;
  CheckData check_data;

  FDTensor x;

  std::vector<float> result = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Linspace(0, 10, 11, &x);
  check_shape({11}, x.Shape());
  check_data(reinterpret_cast<const float*>(x.Data()), result.data(),
             result.size());

  result = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  Linspace(10, 0, 11, &x);
  check_shape({11}, x.Shape());
  check_data(reinterpret_cast<const float*>(x.Data()), result.data(),
             result.size());

  result = {10};
  Linspace(10, 0, 1, &x);
  check_shape({1}, x.Shape());
  check_data(reinterpret_cast<const float*>(x.Data()), result.data(),
             result.size());
}

}  // namespace function
}  // namespace fastdeploy