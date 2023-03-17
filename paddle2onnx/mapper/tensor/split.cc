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

#include "paddle2onnx/mapper/tensor/split.h"

namespace paddle2onnx {
REGISTER_MAPPER(split, SplitMapper)

int32_t SplitMapper::GetMinOpset(bool verbose) {
  int64_t axis = axis_;
  if (HasInput("AxisTensor")) {
    std::vector<int64_t> value;
    if (!TryGetInputValue("AxisTensor", &value)) {
      Error() << "While AxisTensor as the input and it's not a constant "
                 "tensor, the conversion is not supported yet."
              << std::endl;
      return -1;
    }
    axis = value[0];
  }

  if (HasInput("SectionsTensorList")) {
    Logger(verbose, 13) << "While has input SectionsTensorList, "
                        << RequireOpset(13) << std::endl;
    return 13;
  }

  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] < 0) {
      auto info = GetInput("X");
      if (info[0].shape[axis] < 0) {
        Error() << "Cannot convert split op, while there's -1 in sections and "
                   "cannot be infered by input shape."
                << std::endl;
        return -1;
      }
    }
  }
  return 7;
}

void SplitMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  int64_t axis = axis_;
  if (HasInput("AxisTensor")) {
    std::vector<int64_t> value;
    Assert(TryGetInputValue("AxisTensor", &value),
           "[Paddle2ONNX](split) Cannot get constant value from AxisTensor.");
    axis = value[0];
  }
  if (axis < 0) {
    axis += input_info[0].Rank();
  }
  Assert(!HasInput("SectionsTensorList"),
         "[Paddle2ONNX](split) While SectionTensorList as input, requires "
         "opset_version >= 13.");

  int sum_of_kown_dim = 0;
  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] > 0) {
      sum_of_kown_dim += sections_[i];
    }
  }
  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] < 0) {
      Assert(input_info[0].shape[axis] > 0,
             "Cannot convert split op, while there's -1 in sections and cannot "
             "be infered by input shape.");
      sections_[i] = input_info[0].shape[axis] - sum_of_kown_dim;
    }
  }

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }

  helper_->Split(input_info[0].name, output_names, sections_, axis);
}

void SplitMapper::Opset13() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  int64_t axis = axis_;
  if (HasInput("AxisTensor")) {
    std::vector<int64_t> value;
    Assert(TryGetInputValue("AxisTensor", &value),
           "[Paddle2ONNX](split) Cannot get constant value from AxisTensor.");
    axis = value[0];
  }
  if (axis < 0) {
    axis += input_info[0].Rank();
  }

  std::string splits = "";
  if (HasInput("SectionsTensorList")) {
    auto info = GetInput("SectionsTensorList");
    splits = helper_->ConcatIndices(info);
  } else if (sections_.size() > 0) {
    int sum_of_kown_dim = 0;
    for (size_t i = 0; i < sections_.size(); ++i) {
      if (sections_[i] > 0) {
        sum_of_kown_dim += sections_[i];
      }
    }
    for (size_t i = 0; i < sections_.size(); ++i) {
      if (sections_[i] < 0) {
        Assert(input_info[0].shape[axis] > 0,
               "Cannot convert split op, while there's -1 in sections and "
               "cannot be infered by input shape.");
        sections_[i] = input_info[0].shape[axis] - sum_of_kown_dim;
      }
    }
    splits = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, sections_);
  }

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }
  if (splits != "") {
    auto node =
        helper_->MakeNode("Split", {input_info[0].name, splits}, output_names);
    AddAttribute(node, "axis", axis);
  } else {
    auto node = helper_->MakeNode("Split", {input_info[0].name}, output_names);
    AddAttribute(node, "axis", axis);
  }
}

}  // namespace paddle2onnx
