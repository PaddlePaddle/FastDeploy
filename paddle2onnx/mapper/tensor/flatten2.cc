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

#include "paddle2onnx/mapper/tensor/flatten2.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(flatten2, Flatten2Mapper)

int32_t Flatten2Mapper::GetMinOpset(bool verbose) {
  if (GetInput("X")[0].dtype != P2ODataType::FP32 || GetInput("X")[0].dtype != P2ODataType::FP64) {
    Logger(verbose, 9) << "While data type of input is not float32/float64, "<< RequireOpset(9) << std::endl;
    return 9;
  }
  return 7;
}

void Flatten2Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  int64_t axis = axis_;
  auto node = helper_->MakeNode("Flatten", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "axis", axis);
}

}  // namespace paddle2onnx
