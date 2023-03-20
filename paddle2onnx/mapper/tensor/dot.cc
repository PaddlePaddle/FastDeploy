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

#include "paddle2onnx/mapper/tensor/dot.h"

namespace paddle2onnx {
REGISTER_MAPPER(dot, DotMapper)

void DotMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  auto mul_node =
      helper_->MakeNode("Mul", {input_x_info[0].name, input_y_info[0].name});

  if (helper_->GetOpsetVersion() >= 13) {
    std::string axes_node = helper_->Constant(
        {1}, GetOnnxDtype(P2ODataType::INT64), input_x_info[0].Rank() - 1);
    helper_->MakeNode("ReduceSum", {mul_node->output(0), axes_node},
                      {output_info[0].name});
  } else {
    auto reducesum_node = helper_->MakeNode("ReduceSum", {mul_node->output(0)},
                                            {output_info[0].name});
    std::vector<int64_t> axes = {input_x_info[0].Rank() - 1};
    AddAttribute(reducesum_node, "axes", axes);
  }
}

}  // namespace paddle2onnx
