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

#include "paddle2onnx/mapper/tensor/unsqueeze2.h"

namespace paddle2onnx {
REGISTER_MAPPER(unsqueeze2, Unsqueeze2Mapper)

int32_t Unsqueeze2Mapper::GetMinOpset(bool verbose) {
  if (axes_.size() == 0) {
    if (HasInput("AxesTensorList")) {
      Logger(verbose, 13) << "While AxisTensorList as input, "
                          << RequireOpset(13) << std::endl;
      return 13;
    } else if (HasInput("AxesTensor")) {
      auto info = GetInput("AxesTensor");
      if (!IsConstantInput("AxesTensor")) {
        Logger(verbose, 13)
            << "While AxesTensor as input, and it's not a constant tensor, "
            << RequireOpset(13) << std::endl;
        return 13;
      } else {
        return 7;
      }
    }
  }
  return 7;
}

void Unsqueeze2Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<int64_t> axes;
  if (axes_.empty()) {
    Assert(TryGetInputValue("AxesTensor", &axes),
           "While unsqueeze2 has input AxesTensor, it cannot be exported by "
           "Paddle2ONNX");
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] = axes[i] + input_info[0].Rank() + i + 1;
    }
  }
  helper_->Unsqueeze(input_info[0].name, output_info[0].name, axes);
}

void Unsqueeze2Mapper::Opset13() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<int64_t> axes;
  if (axes_.empty()) {
    TryGetInputValue("AxesTensor", &axes);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] = axes[i] + input_info[0].Rank() + i + 1;
    }
  }

  if (axes.size() > 0) {
    helper_->Unsqueeze(input_info[0].name, output_info[0].name, axes);
  } else {
    std::string axes_node = "";
    if (HasInput("AxesTensorList")) {
      auto info = GetInput("AxesTensorList");
      axes_node = helper_->ConcatIndices(info);
    } else {
      auto info = GetInput("AxesTensor");
      axes_node =
          helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
    }
    helper_->MakeNode("Unsqueeze", {input_info[0].name, axes_node},
                      {output_info[0].name});
  }
}

}  // namespace paddle2onnx
