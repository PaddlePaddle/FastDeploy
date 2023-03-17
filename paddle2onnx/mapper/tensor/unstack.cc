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

#include "paddle2onnx/mapper/tensor/unstack.h"

namespace paddle2onnx {
REGISTER_MAPPER(unstack, UnstackMapper)

void UnstackMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetOutput("Y");

  if (axis_ < 0) {
    axis_ = axis_ + x_info[0].Rank();
  }

  auto split_nodes = helper_->Split(
      x_info[0].name, std::vector<int64_t>(y_info.size(), 1), axis_);

  for (size_t i = 0; i < split_nodes.size(); ++i) {
    helper_->Squeeze(split_nodes[i], y_info[i].name, {axis_});
  }
}

}  // namespace paddle2onnx
