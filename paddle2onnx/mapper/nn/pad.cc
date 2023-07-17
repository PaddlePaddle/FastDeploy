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

#include "paddle2onnx/mapper/nn/pad.h"

namespace paddle2onnx {
REGISTER_MAPPER(pad, PadMapper)

std::vector<int64_t> PadMapper::ConvertPaddingParameter(
    const std::vector<int64_t>& paddings) {
  std::vector<int64_t> new_paddings(paddings.size(), 0);
  Assert(paddings.size() % 2 == 0, "The size of padding should be even");
  int64_t half_paddings_len = paddings.size() / 2;
  for (auto i = 0; i < half_paddings_len; ++i) {
    new_paddings[i] = paddings[2 * i];
    new_paddings[i + half_paddings_len] = paddings[2 * i + 1];
  }
  return new_paddings;
}

void PadMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node =
      helper_->MakeNode("Pad", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "mode", "constant");
  AddAttribute(node, "value", pad_value_);
  AddAttribute(node, "pads", ConvertPaddingParameter(paddings_));
}

void PadMapper::Opset11() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto paddings = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                    ConvertPaddingParameter(paddings_));
  auto value =
      helper_->Constant({}, GetOnnxDtype(input_info[0].dtype), pad_value_);
  auto node = helper_->MakeNode("Pad", {input_info[0].name, paddings, value},
                                {output_info[0].name});
  AddAttribute(node, "mode", "constant");
}

}  // namespace paddle2onnx
