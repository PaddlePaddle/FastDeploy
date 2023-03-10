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

#include "paddle2onnx/mapper/tensor/add_n.h"

namespace paddle2onnx {
REGISTER_MAPPER(sum, AddNMapper)

void AddNMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  if (x_info.size() == 1) {
    helper_->AutoCast(x_info[0].name, out_info[0].name, x_info[0].dtype,
                      out_info[0].dtype);
  } else {
    std::vector<std::string> inputs;
    for (auto i = 0; i < x_info.size(); ++i) {
      inputs.push_back(helper_->AutoCast(x_info[i].name, x_info[0].dtype,
                                         P2ODataType::FP32));
    }
    auto output = helper_->MakeNode("Sum", inputs)->output(0);
    helper_->AutoCast(output, out_info[0].name, P2ODataType::FP32,
                      out_info[0].dtype);
  }
}

}  // namespace paddle2onnx
