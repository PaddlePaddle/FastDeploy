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
#pragma once

#include <vector>
#include "gtest/gtest.h"

namespace fastdeploy {

struct CheckShape {
  template <typename T>
  void operator()(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (int i = 0; i < lhs.size(); ++i) {
      ASSERT_EQ(lhs[i], rhs[i]);
    }
  }
};

struct CheckData {
  template <typename T>
  void operator()(const T* lhs_ptr, const T* rhs_ptr, int num) {
    for (int i = 0; i < num; ++i) {
      ASSERT_EQ(lhs_ptr[i], rhs_ptr[i]);
    }
  }
  void operator()(const float* lhs_ptr, const float* rhs_ptr, int num) {
    for (int i = 0; i < num; ++i) {
      ASSERT_FLOAT_EQ(lhs_ptr[i], rhs_ptr[i]);
    }
  }
  void operator()(const double* lhs_ptr, const double* rhs_ptr, int num) {
    for (int i = 0; i < num; ++i) {
      ASSERT_DOUBLE_EQ(lhs_ptr[i], rhs_ptr[i]);
    }
  }
};

}  // namespace fastdeploy
