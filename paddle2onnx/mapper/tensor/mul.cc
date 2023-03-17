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

#include "paddle2onnx/mapper/tensor/mul.h"

namespace paddle2onnx {
REGISTER_MAPPER(mul, MulMapper)

void MulMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto out_info = GetOutput("Out");

  auto x_input = x_info[0].name;
  auto y_input = y_info[0].name;
  if (x_info[0].Rank() > 2) {
    auto node = helper_->MakeNode("Flatten", {x_input});
    AddAttribute(node, "axis", x_num_col_dims_);
    x_input = node->output(0);
  }
  if (y_info[0].Rank() > 2) {
    auto node = helper_->MakeNode("Flatten", {y_input});
    AddAttribute(node, "axis", y_num_col_dims_);
    y_input = node->output(0);
  }
  auto out = helper_->MakeNode("MatMul", {x_input, y_input})->output(0);

  if (x_info[0].Rank() != 2 || y_info[0].Rank() != 2) {
    auto x_shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
    auto y_shape = helper_->MakeNode("Shape", {y_info[0].name})->output(0);
    auto out_shape_0 = helper_->Slice(x_shape, {0}, {0}, {x_num_col_dims_});
    auto out_shape_1 = helper_->Slice(y_shape, {0}, {y_num_col_dims_}, {y_info[0].Rank()});
    auto out_shape = helper_->Concat({out_shape_0, out_shape_1}, 0);
    out = helper_->MakeNode("Reshape", {out, out_shape})->output(0);
  }
  helper_->MakeNode("Identity", {out}, {out_info[0].name});
}

}  // namespace paddle2onnx
