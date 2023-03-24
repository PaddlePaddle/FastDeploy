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

#include "paddle2onnx/mapper/tensor/assign.h"

namespace paddle2onnx {
REGISTER_MAPPER(assign, AssignMapper)
REGISTER_MAPPER(share_data, AssignMapper)

void AssignMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (block_idx_ != 0 && OpType() != "share_data") {
    // Here's a trick for tensorrt
    // Consider remove this trick
    if (input_info[0].dtype == P2ODataType::BOOL) {
      auto zero = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                    std::vector<int64_t>(1, 0));
      auto cast_input = helper_->AutoCast(input_info[0].name, P2ODataType::BOOL,
                                          P2ODataType::INT64);
      auto result = helper_->MakeNode("Add", {cast_input, zero})->output(0);
      helper_->AutoCast(result, output_info[0].name, P2ODataType::INT64,
                        output_info[0].dtype);
    } else {
      auto zero = helper_->Constant(GetOnnxDtype(input_info[0].dtype),
                                    std::vector<double>(1, 0.0));
      auto new_input =
          helper_->Unsqueeze(input_info[0].name, std::vector<int64_t>(1, 0));
      auto result = helper_->MakeNode("Add", {new_input, zero})->output(0);
      helper_->Squeeze(result, output_info[0].name, std::vector<int64_t>(1, 0));
    }
  } else {
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
