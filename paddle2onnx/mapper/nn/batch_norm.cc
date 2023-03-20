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

#include "paddle2onnx/mapper/nn/batch_norm.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(batch_norm, BatchNormMapper)

void BatchNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto scale_info = GetInput("Scale");
  auto bias_info = GetInput("Bias");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");

  auto node = helper_->MakeNode(
      "BatchNormalization",
      {input_info[0].name, scale_info[0].name, bias_info[0].name,
       mean_info[0].name, variance_info[0].name},
      {output_info[0].name});
  if (helper_->GetOpsetVersion() < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);
}

}  // namespace paddle2onnx
