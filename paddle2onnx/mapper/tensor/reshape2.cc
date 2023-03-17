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

#include "paddle2onnx/mapper/tensor/reshape2.h"

#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(reshape2, Reshape2Mapper)

void Reshape2Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string shape_name = "ShapeTensor";
  if (!HasInput(shape_name)) {
    shape_name = "Shape";
  }

  std::string new_shape = "";
  if (HasInput(shape_name)) {
    auto shape_info = GetInput(shape_name);
    if (shape_info.size() > 1) {
      new_shape = helper_->ConcatIndices(shape_info);
    } else {
      new_shape = helper_->AutoCast(shape_info[0].name, shape_info[0].dtype,
                                    P2ODataType::INT64);
    }
  } else {
    std::vector<int64_t> value;
    GetAttr("shape", &value);
    new_shape = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, value);
  }
  auto node = helper_->MakeNode("Reshape", {input_info[0].name, new_shape},
                    {output_info[0].name});
  if (helper_->GetOpsetVersion()>= 14) {
    AddAttribute(node, "allowzero", int64_t(0));
  }
}

}  // namespace paddle2onnx
