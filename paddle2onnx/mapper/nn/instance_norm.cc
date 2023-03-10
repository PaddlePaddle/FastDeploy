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

#include "paddle2onnx/mapper/nn/instance_norm.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(instance_norm, InstanceNormMapper)

int32_t InstanceNormMapper::GetMinOpset(bool verbose) {
  auto input_info = GetInput("X");
  int num_groups = input_info[0].shape[1];
  if (num_groups < 0) {
    Error() << "The dimension in axis=1 of input tensor must be known, but now it's unknown." << std::endl;
    return -1;
  }
  return 7;
}

void InstanceNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Y");
  int num_groups = input_info[0].shape[1];

  std::string scale = "";
  if (HasInput("Scale")) {
    scale = GetInput("Scale")[0].name;
  } else {
    scale = helper_->Constant(GetOnnxDtype(input_info[0].dtype), std::vector<float>(num_groups, 1.0));
  }
  
  std::string bias = "";
  if (HasInput("Bias")) {
    bias = GetInput("Bias")[0].name;
  } else {
    bias = helper_->Constant(GetOnnxDtype(input_info[0].dtype), std::vector<float>(num_groups, 0.0));
  }

  auto node = helper_->MakeNode(
      "InstanceNormalization",
      {input_info[0].name, scale, bias},
      {output_info[0].name});
  AddAttribute(node, "epsilon", epsilon_);
}

}  // namespace paddle2onnx
