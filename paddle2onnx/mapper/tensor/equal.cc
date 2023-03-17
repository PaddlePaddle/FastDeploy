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

#include "paddle2onnx/mapper/tensor/equal.h"

namespace paddle2onnx {
REGISTER_MAPPER(equal, EqualMapper)

int32_t EqualMapper::GetMinOpset(bool verbose) {
  if (axis_ != -1) {
    Error() << "axis attribute must be -1 in operator equal." << std::endl;
    return -1;
  }
  return 7;
}

void EqualMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  std::string input_x = input_x_info[0].name;
  std::string input_y = input_y_info[0].name;
  if (helper_->GetOpsetVersion() < 11) {
    input_x = helper_->AutoCast(input_x_info[0].name, input_x_info[0].dtype,
                                P2ODataType::INT32);
    input_y = helper_->AutoCast(input_y_info[0].name, input_y_info[0].dtype,
                                P2ODataType::INT32);
  }
  helper_->MakeNode("Equal", {input_x, input_y}, {output_info[0].name});
}

}  // namespace paddle2onnx
