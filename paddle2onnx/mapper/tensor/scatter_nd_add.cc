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

#include "paddle2onnx/mapper/tensor/scatter_nd_add.h"

namespace paddle2onnx {
REGISTER_MAPPER(scatter_nd_add, ScatterNdAddMapper)

int32_t ScatterNdAddMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 16) << RequireOpset(16) << std::endl;
  return 16;
}

void ScatterNdAddMapper::Opset16() {
  auto input_x_info = GetInput("X");
  auto input_ids_info = GetInput("Index");
  auto input_updates_info = GetInput("Updates");
  auto output_info = GetOutput("Out");

  auto shape_node = helper_->MakeNode("Shape", {input_x_info[0].name});

  std::string zeros_like_node = helper_->ConstOfShape(
      shape_node->output(0), GetOnnxDtype(input_x_info[0].dtype),
      static_cast<float>(0));

  std::string input_ids_node = helper_->AutoCast(
      input_ids_info[0].name, input_ids_info[0].dtype, P2ODataType::INT64);

  auto scatter_nd_node = helper_->MakeNode(
      "ScatterND",
      {zeros_like_node, input_ids_node, input_updates_info[0].name});
  AddAttribute(scatter_nd_node, "reduction", "add");
  helper_->MakeNode("Add", {input_x_info[0].name, scatter_nd_node->output(0)},
                    {output_info[0].name});
}

}  // namespace paddle2onnx
