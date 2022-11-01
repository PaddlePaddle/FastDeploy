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
#include <cmath>
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
  void operator()(const T* lhs_ptr, const T* rhs_ptr, int num, int atol = 0) {
    for (int i = 0; i < num; ++i) {
//      ASSERT_FLOAT_EQ(lhs_ptr[i], rhs_ptr[i]);
      int abs_diff = abs(lhs_ptr[i] - rhs_ptr[i]);
      if (abs_diff > atol) {
        std::cout << "lhs_ptr: " << static_cast<int64_t>(lhs_ptr[i])
                  << " rhs_ptr: " << static_cast<int64_t>(rhs_ptr[i])
                  << " abs_diff: " << abs_diff << std::endl;
        ASSERT_EQ(1, 0);
      }
      ASSERT_EQ(1, 1);
    }
  }
  void operator()(const float* lhs_ptr, const float* rhs_ptr,
                int num, float atol = 1e-06, float rtol = 1e-06) {
    for (int i = 0; i < num; ++i) {
      float abs_diff = fabs(lhs_ptr[i] - rhs_ptr[i]);
      float rel_diff = abs_diff / (std::max(fabs(lhs_ptr[i]),
                            fabs(rhs_ptr[i])) + 1e-06);
      if (abs_diff > atol && rel_diff > rtol) {
        std::cout << "lhs_ptr: " << lhs_ptr[i] << " rhs_ptr: "
                  << rhs_ptr[i] << " abs_diff: " << abs_diff
                  << " rel_diff: " << rel_diff << std::endl;
        ASSERT_EQ(1, 0);
      }
      ASSERT_EQ(1, 1);
    }
  }

  void operator()(const double* lhs_ptr, const double* rhs_ptr, int num) {
    for (int i = 0; i < num; ++i) {
      ASSERT_DOUBLE_EQ(lhs_ptr[i], rhs_ptr[i]);
    }
  }
};

struct CheckType {
  void operator()(FDDataType type1, FDDataType type2) {
    ASSERT_EQ(type1, type2);
  }
};

}  // namespace fastdeploy
