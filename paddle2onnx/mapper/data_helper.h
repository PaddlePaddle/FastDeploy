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
#include "paddle2onnx/utils/utils.h"

namespace paddle2onnx {

inline std::vector<int64_t> Arange(int64_t start, int64_t end) {
  Assert(end > start, "In arrange(), end must be greater than start.");
  std::vector<int64_t> res;
  res.resize(end - start);
  for (auto i = start; i < end; ++i) {
    res[i - start] = i;
  }
  return res;
}
}  // namespace paddle2onnx
