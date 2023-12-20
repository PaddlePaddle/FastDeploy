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

#include "paddle2onnx/mapper/tensor/argmin.h"

namespace paddle2onnx {
REGISTER_MAPPER(arg_min, ArgMinMapper)

int32_t ArgMinMapper::GetMinOpset(bool verbose) {
  if (IsAttrVar("axis") && !IsConstant(GetAttrVar("axis")[0])) {
    Error() << "While Attribute(axis)'s type is Tensor, it's not "
               "supported "
               "unless it's a constant tensor."
            << std::endl;
    return -1;
  }
  return 7;
}

void ArgMinMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto input = input_info[0].name;
  if (flatten_) {
    input = helper_->Flatten(input_info[0].name);
  }

  if (IsAttrVar("axis")) {
    auto axis_info = GetAttrVar("axis");
    std::vector<int64_t> temp;
    TryGetValue(axis_info[0], &temp);
    axis_ = temp[0];
  } else {
    GetAttr("axis", &axis_);
  }

  if (input_info[0].dtype == P2ODataType::FP64) {
    input = helper_->AutoCast(input, P2ODataType::FP64, P2ODataType::FP32);
  }
  if (input_info[0].dtype == P2ODataType::INT64) {
    input = helper_->AutoCast(input, P2ODataType::INT64, P2ODataType::INT32);
  }
  auto arg_node = helper_->MakeNode("ArgMin", {input});
  AddAttribute(arg_node, "axis", axis_);
  AddAttribute(arg_node, "keepdims", static_cast<int64_t>(keepdims_));
  if (keepdims_) {
    std::vector<int64_t> shape(input_info[0].Rank(), 1);
    std::string out = arg_node->output(0);
    if (flatten_) {
      out = helper_->Reshape(arg_node->output(0), shape);
    }
    helper_->AutoCast(out, output_info[0].name, P2ODataType::INT64,
                      output_info[0].dtype);
  } else {
    helper_->AutoCast(arg_node->output(0), output_info[0].name,
                      P2ODataType::INT64, output_info[0].dtype);
  }
}

}  // namespace paddle2onnx
