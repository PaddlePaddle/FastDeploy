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

#include "paddle2onnx/mapper/nn/pad3d.h"

namespace paddle2onnx {
REGISTER_MAPPER(pad3d, Pad3DMapper)

int32_t Pad3DMapper::GetMinOpset(bool verbose) {
  if (data_format_ == "NDHWC") {
    Error() << "NDHWC format is not supported." << std::endl;
    return -1;
  }
  if (mode_ == "circular") {
    Error() << "Padding mode `circular` is not supported." << std::endl;
    return -1;
  }
  if (HasInput("Paddings")) {
    if (!IsConstantInput("Paddings")) {
      Logger(verbose, 11) << "While Paddings is input and it's not a constant tensor, " << RequireOpset(11) << std::endl;
      return 11;
    }
    std::vector<int64_t> paddings;
    if (!TryGetInputValue("Paddings", &paddings)) {
      Logger(verbose, 11) << "Cannot get constant value from input of Paddings, " << RequireOpset(11) << std::endl;
      return 11;
    } else {
      if (paddings.size() != 6) {
       Error() << "Size of paddings should be equal to 6, but now it's " << paddings.size() << std::endl;
       return -1;
      }
    }
  } else {
    if (paddings_.size() != 6) {
      Error() << "Size of paddings should be equal to 6, but now it's " << paddings_.size() << std::endl;
      return -1;
    }
  }
  return 7;
}

std::vector<int64_t> Pad3DMapper::ConvertPaddingParameter(const std::vector<int64_t>& paddings) {
  std::vector<int64_t> new_paddings(10, 0);
  new_paddings[2] = paddings[4];
  new_paddings[3] = paddings[2];
  new_paddings[4] = paddings[0];
  new_paddings[7] = paddings[5];
  new_paddings[8] = paddings[3];
  new_paddings[9] = paddings[1];
  return new_paddings;
}

void Pad3DMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto mode = mode_;
  if (mode == "replicate") {
    mode = "edge";
  }
  std::vector<int64_t> paddings;
  if (HasInput("Paddings")) {
    Assert(TryGetInputValue("Paddings", &paddings), "Cannot get constant value from input of Paddings, " + RequireOpset(11));
  } else {
    paddings.assign(paddings_.begin(), paddings_.end());
  }
  std::vector<int64_t> new_paddings = ConvertPaddingParameter(paddings);
  auto node = helper_->MakeNode("Pad", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "mode", mode);
  AddAttribute(node, "value", value_);
  AddAttribute(node, "pads", new_paddings);
}

void Pad3DMapper::Opset11() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto mode = mode_;
  if (mode == "replicate") {
    mode = "edge";
  }

  std::string paddings = "";
  if (HasInput("Paddings")) {
    std::vector<int64_t> paddings_value;
    if (TryGetInputValue("Paddings", &paddings_value)) {
      std::vector<int64_t> new_paddings = ConvertPaddingParameter(paddings_value);
      paddings = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, new_paddings);
    } else {
      auto pad_info = GetInput("Paddings");
      auto cast_pad = helper_->AutoCast(pad_info[0].name, pad_info[0].dtype, P2ODataType::INT64);
      auto split_pads = helper_->Split(cast_pad, std::vector<int64_t>(6, 1), 0);
      auto zero = helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(0));
      paddings = helper_->Concat({zero, zero, split_pads[4], split_pads[2], split_pads[0], zero, zero, split_pads[5], split_pads[3], split_pads[1]}, 0);
    }
  } else {
    std::vector<int64_t> new_paddings = ConvertPaddingParameter(paddings_);
    paddings = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, new_paddings);
  }
  auto value = helper_->Constant({}, GetOnnxDtype(input_info[0].dtype), value_);
  auto node = helper_->MakeNode("Pad", {input_info[0].name, paddings, value}, {output_info[0].name});
  AddAttribute(node, "mode", mode);
}

}  // namespace paddle2onnx
