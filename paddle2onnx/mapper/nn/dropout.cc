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
#include "paddle2onnx/mapper/nn/dropout.h"

#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(dropout, DropoutMapper)

int32_t DropoutMapper::GetMinOpset(bool verbose) {
  if (dropout_implementation_ != "downgrade_in_infer" &&
      dropout_implementation_ != "upscale_in_train") {
    Error() << "Drop out type: " << dropout_implementation_
            << " is not supported yet." << std::endl;
    return -1;
  }
  if (dropout_implementation_ == "downgrade_in_infer") {
    if (IsAttrVar("dropout_prob") &&
        !IsConstant(GetAttrVar("dropout_prob")[0])) {
      Error() << "While Attribute(dropout_prob)'s type is Tensor, it's not "
                 "supported "
                 "unless it's a constant tensor when dropout_implementation is "
                 "downgrade_in_infer."
              << std::endl;
      return -1;
    }
  }
  return 7;
}

void DropoutMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  if (dropout_implementation_ == "upscale_in_train") {
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    if (IsAttrVar("dropout_prob")) {
      auto prob_info = GetAttrVar("dropout_prob");
      std::vector<float> temp;
      TryGetValue(prob_info[0], &temp);
      dropout_prob_ = temp[0];
    } else {
      GetAttr("dropout_prob", &dropout_prob_);
    }
    std::string scale_node = helper_->Constant(
        {}, GetOnnxDtype(input_info[0].dtype), 1 - dropout_prob_);
    helper_->MakeNode("Mul", {input_info[0].name, scale_node},
                      {output_info[0].name});
  }
}

}  // namespace paddle2onnx
