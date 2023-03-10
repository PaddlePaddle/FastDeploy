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

#include "paddle2onnx/mapper/tensor/argsort.h"

namespace paddle2onnx {
REGISTER_MAPPER(argsort, ArgsortMapper)

int32_t ArgsortMapper::GetMinOpset(bool verbose) {
  if (!descending_) {
    Logger(verbose, 11) << "While descending=False, " << RequireOpset(11)
                        << std::endl;
    return 11;
  }

  if (axis_ < 0) {
    axis_ = axis_ + GetInput("X")[0].Rank();
  }
  if (GetInput("X")[0].shape[axis_] <= 0) {
    Logger(verbose, 10) << "While input shape is dynamic, " << RequireOpset(10)
                        << std::endl;
    return 10;
  }
  return 7;
}

void ArgsortMapper::Opset10() {
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  auto shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
  if (axis_ < 0) {
    axis_ = axis_ + x_info[0].Rank();
  }
  auto dim_size = helper_->Slice(shape, {0}, {axis_}, {axis_ + 1});

  auto out_node =
      helper_->MakeNode("TopK", {x_info[0].name, dim_size},
                        {output_info[0].name, indices_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  if (helper_->GetOpsetVersion() > 10) {
    if (!descending_) {
      AddAttribute(out_node, "largest", static_cast<int64_t>(0));
    } else {
      AddAttribute(out_node, "largest", static_cast<int64_t>(1));
    }
  }
}

void ArgsortMapper::Opset7() {
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  if (axis_ < 0) {
    axis_ = axis_ + x_info[0].Rank();
  }

  auto out_node = helper_->MakeNode(
      "TopK", {x_info[0].name}, {output_info[0].name, indices_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  AddAttribute(out_node, "k", x_info[0].shape[axis_]);
}

}  // namespace paddle2onnx
