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

#include "paddle2onnx/mapper/tensor/scatter.h"

namespace paddle2onnx {
REGISTER_MAPPER(scatter, ScatterMapper)

int32_t ScatterMapper::GetMinOpset(bool verbose) {
  if (!overwrite_) {
    Logger(verbose, 16) << "When overwrite is False, " << RequireOpset(16)
                        << std::endl;
    return 16;
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void ScatterMapper::Opset11() {
  auto input_x_info = GetInput("X");
  auto input_ids_info = GetInput("Ids");
  auto input_updates_info = GetInput("Updates");
  auto output_info = GetOutput("Out");

  std::string ids_node = helper_->AutoCast(
      input_ids_info[0].name, input_ids_info[0].dtype, P2ODataType::INT64);

  std::string shape_node;
  if (input_ids_info[0].Rank() == 0) {
    std::vector<int64_t> shape = {1};
    shape_node = helper_->Constant(GetOnnxDtype(P2ODataType::INT64), shape);
  } else {
    std::vector<int64_t> shape = {input_ids_info[0].shape[0], 1};
    shape_node = helper_->Constant(GetOnnxDtype(P2ODataType::INT64), shape);
  }

  auto reshape_index_node =
      helper_->MakeNode("Reshape", {ids_node, shape_node});

  if (!overwrite_) {
    auto shape_node = helper_->MakeNode("Shape", {input_x_info[0].name});
    std::string zeros_like_node = helper_->ConstOfShape(
        shape_node->output(0), GetOnnxDtype(input_x_info[0].dtype),
        static_cast<float>(0));
    auto scatter_nd_node = helper_->MakeNode(
        "ScatterND", {zeros_like_node, reshape_index_node->output(0),
                      input_updates_info[0].name});
    AddAttribute(scatter_nd_node, "reduction", "add");

    std::string zero_node = helper_->Constant(
        {}, GetOnnxDtype(input_x_info[0].dtype), static_cast<float>(0));

    auto equal_node =
        helper_->MakeNode("Equal", {scatter_nd_node->output(0), zero_node});

    std::string condition_node = helper_->AutoCast(
        equal_node->output(0), P2ODataType::INT64, P2ODataType::BOOL);

    helper_->MakeNode(
        "Where",
        {condition_node, input_x_info[0].name, scatter_nd_node->output(0)},
        {output_info[0].name});
  } else {
    auto node =
        helper_->MakeNode("ScatterND",
                          {input_x_info[0].name, reshape_index_node->output(0),
                           input_updates_info[0].name},
                          {output_info[0].name});
  }
}

}  // namespace paddle2onnx
