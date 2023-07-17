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

#include "paddle2onnx/mapper/tensor/cumsum.h"

namespace paddle2onnx {
REGISTER_MAPPER(cumsum, CumsumMapper)

int32_t CumsumMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void CumsumMapper::Opset11() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].Rank() == 0) {
    auto axis_node = helper_->Constant({}, GetOnnxDtype(P2ODataType::INT64), 0);
    auto unsqueeze_node = helper_->Unsqueeze(input_info[0].name, {0});
    auto cumsum_node = helper_->MakeNode("CumSum", {unsqueeze_node, axis_node});
    if (flatten_) {
      helper_->AutoCast(cumsum_node->output(0), output_info[0].name,
                        input_info[0].dtype, output_info[0].dtype);
    } else {
      helper_->Squeeze(cumsum_node->output(0), output_info[0].name, {0});
    }
  } else {
    std::string axis_node;
    if (IsAttrVar("axis")) {
      auto axis_info = GetAttrVar("axis");
      axis_node = helper_->AutoCast(axis_info[0].name, axis_info[0].dtype,
                                    P2ODataType::INT64);
    } else {
      GetAttr("axis", &axis_);
      axis_node =
          helper_->Constant({}, GetOnnxDtype(P2ODataType::INT64), axis_);
    }
    std::string input_node = input_info[0].name;
    if (flatten_) {
      input_node = helper_->Reshape(input_info[0].name, {-1});
    }
    helper_->MakeNode("CumSum", {input_node, axis_node}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
