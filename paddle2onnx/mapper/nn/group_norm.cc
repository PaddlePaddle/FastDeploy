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

#include "paddle2onnx/mapper/nn/group_norm.h"

#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(group_norm, GroupNormMapper)

int32_t GroupNormMapper::GetMinOpset(bool verbose) {
  auto input_info = GetInput("X");
  if (input_info[0].Rank() != 4) {
    Error() << "Only support 4D-Tensor as input for GroupNorm" << std::endl;
    return -1;
  }
  return 7;
}

void GroupNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Y");

  std::vector<int64_t> shape_val = {0, groups_, -1};
  std::string shape =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64), shape_val);

  auto reshape_input =
      helper_->MakeNode("Reshape", {input_info[0].name, shape});

  std::string scale_ = helper_->Constant(GetOnnxDtype(input_info[0].dtype),
                                         std::vector<float>(groups_, 1.0));

  std::string bias_ = helper_->Constant(GetOnnxDtype(input_info[0].dtype),
                                        std::vector<float>(groups_, 0.0));

  auto reshaped_output = helper_->MakeNode(
      "InstanceNormalization", {reshape_input->output(0), scale_, bias_});
  AddAttribute(reshaped_output, "epsilon", epsilon_);

  auto origin_shape = helper_->MakeNode("Shape", {input_info[0].name});

  if (HasInput("Scale") && HasInput("Bias")) {
    auto scale_info = GetInput("Scale");
    auto bias_info = GetInput("Bias");
    auto output = helper_->MakeNode(
        "Reshape", {reshaped_output->output(0), origin_shape->output(0)});
    std::string unsqueezed_scale =
        helper_->Unsqueeze(scale_info[0].name, {1, 2});
    std::string unsqueezed_bias = helper_->Unsqueeze(bias_info[0].name, {1, 2});
    auto scale_output =
        helper_->MakeNode("Mul", {output->output(0), unsqueezed_scale});
    helper_->MakeNode("Add", {scale_output->output(0), unsqueezed_bias},
                      {output_info[0].name});
  } else {
    helper_->MakeNode("Reshape",
                      {reshaped_output->output(0), origin_shape->output(0)},
                      {output_info[0].name});
  }
}

}  // namespace paddle2onnx
