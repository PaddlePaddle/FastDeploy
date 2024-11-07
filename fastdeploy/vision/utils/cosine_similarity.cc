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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace utils {

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b,
                       bool normalized) {
  FDASSERT((a.size() == b.size()) && (a.size() != 0),
           "The size of a and b must be equal and >= 1.");
  size_t num_val = a.size();
  float mul_ab = 0.f;
  if (normalized) {
    for (size_t i = 0; i < num_val; ++i) {
      mul_ab += (a[i] * b[i]);
    }
    return mul_ab;
  }
  auto norm_a = L2Normalize(a);
  auto norm_b = L2Normalize(b);
  for (size_t i = 0; i < num_val; ++i) {
    mul_ab += (norm_a[i] * norm_b[i]);
  }
  return mul_ab;
}

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
