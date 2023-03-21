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

#include "paddle2onnx/mapper/nn/affine_channel.h"

namespace paddle2onnx {
REGISTER_MAPPER(affine_channel, AffineChannelMapper)

int32_t AffineChannelMapper::GetMinOpset(bool verbose) {
  if (data_layout_ == "NHWC") {
    Error() << "Data format NHWC is not supported." << std::endl;
    return false;
  }
  return 7;
}

void AffineChannelMapper::Opset7() {
  auto x_info = GetInput("X");
  auto scale_info = GetInput("Scale");
  auto bias_info = GetInput("Bias");
  auto out_info = GetOutput("Out");

  auto scale = scale_info[0].name;
  auto bias = bias_info[0].name;
  if (scale_info[0].shape.size() <= 1) {
    scale = helper_->Reshape(scale, {1, -1, 1, 1});
    bias = helper_->Reshape(bias, {1, -1, 1, 1});
  }
  auto out = helper_->MakeNode("Mul", {x_info[0].name, scale})->output(0);
  helper_->MakeNode("Add", {out, bias}, {out_info[0].name});
}

}  // namespace paddle2onnx
