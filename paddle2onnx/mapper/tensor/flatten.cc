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

#include "paddle2onnx/mapper/tensor/flatten.h"

#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(flatten_contiguous_range, FlattenMapper)

void FlattenMapper::Opset7() {
  auto input_info = GetInput("X");
  if (start_axis_ < 0) {
    start_axis_ += input_info[0].Rank();
  }
  if (stop_axis_ < 0) {
    stop_axis_ += input_info[0].Rank();
  }
  auto output_info = GetOutput("Out");

  auto unknown_dim_node =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, -1);
  if (start_axis_ == 0 && stop_axis_ == input_info[0].Rank() - 1) {
    helper_->MakeNode("Reshape", {input_info[0].name, unknown_dim_node},
                      {output_info[0].name});
  } else {
    auto input_shape_node = helper_->MakeNode("Shape", {input_info[0].name});
    if (start_axis_ == 0) {
      auto second_part_shape =
          helper_->Slice(input_shape_node->output(0), {0}, {stop_axis_ + 1},
                         {input_info[0].Rank()});
      auto new_shape_node =
          helper_->MakeNode("Concat", {unknown_dim_node, second_part_shape});
      AddAttribute(new_shape_node, "axis", int64_t(0));
      helper_->MakeNode("Reshape",
                        {input_info[0].name, new_shape_node->output(0)},
                        {output_info[0].name});
    } else if (stop_axis_ == input_info[0].Rank() - 1) {
      auto first_part_shape =
          helper_->Slice(input_shape_node->output(0), {0}, {0}, {start_axis_});
      auto new_shape_node =
          helper_->MakeNode("Concat", {first_part_shape, unknown_dim_node});
      AddAttribute(new_shape_node, "axis", int64_t(0));
      helper_->MakeNode("Reshape",
                        {input_info[0].name, new_shape_node->output(0)},
                        {output_info[0].name});
    } else {
      auto first_part_shape =
          helper_->Slice(input_shape_node->output(0), {0}, {0}, {start_axis_});
      auto second_part_shape =
          helper_->Slice(input_shape_node->output(0), {0}, {stop_axis_ + 1},
                         {input_info[0].Rank()});
      auto new_shape_node = helper_->MakeNode(
          "Concat", {first_part_shape, unknown_dim_node, second_part_shape});
      AddAttribute(new_shape_node, "axis", int64_t(0));
      helper_->MakeNode("Reshape",
                        {input_info[0].name, new_shape_node->output(0)},
                        {output_info[0].name});
    }
  }
}

}  // namespace paddle2onnx
