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

#include "paddle2onnx/mapper/tensor/slice.h"

#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(slice, SliceMapper)
REGISTER_MAPPER(strided_slice, SliceMapper)

int32_t SliceMapper::GetMinOpset(bool verbose) {
  if (HasInput("StartsTensorList") || HasInput("EndsTensorList") ||
      HasInput("StridesTensorList")) {
    Logger(verbose, 10)
        << "While has input StartsTensorList/EndsTensorListStridesTensorList, "
        << RequireOpset(10) << std::endl;
    return 10;
  }
  if (HasInput("StartsTensor")) {
    auto info = GetInput("StartsTensor");
    if (!IsConstantInput("StartsTensor")) {
      Logger(verbose, 10)
          << "While has input StartsTensor, and it's not a constant tensor, "
          << RequireOpset(10) << std::endl;
      return 10;
    }
  }
  if (HasInput("EndsTensor")) {
    auto info = GetInput("EndsTensor");
    if (!IsConstantInput("EndsTensor")) {
      Logger(verbose, 10)
          << "While has input EndsTensor, and it's not a constant tensor, "
          << RequireOpset(10) << std::endl;
      return 10;
    }
  }
  if (HasInput("StridesTensor") || strides_.size() > 0) {
    Logger(verbose, 10) << "While has strides, " << RequireOpset(10)
                        << std::endl;
    return 10;
  }
  return 7;
}

std::vector<int64_t> SliceMapper::DecreaseAxis() {
  std::vector<int64_t> decrease_axis;
  bool has_attr = HasAttr("decrease_axis");
  if (has_attr) {
    GetAttr("decrease_axis", &decrease_axis);
    auto input_info = GetInput("Input");
    auto output_info = GetOutput("Out");
    if (output_info[0].shape.size() == 1 && output_info[0].shape[0] == 0) {
      return decrease_axis;
    }
    if (input_info[0].shape.size() > output_info[0].shape.size()) {
      return decrease_axis;
    }
    return {};
  }
  return decrease_axis;
}

void SliceMapper::Opset7() {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");

  Assert(!HasInput("StartsTensorList"),
         "While slice/strided_slice has input StartsTensorList, requires "
         "opset_version >= 10");

  std::vector<int64_t> starts;
  if (HasInput("StartsTensor")) {
    Assert(TryGetInputValue("StartsTensor", &starts),
           "While slice/strided_slice has input StartsTensor, and it's not a "
           "constant tensor, then requires opset_version >= 10");
  } else {
    starts = starts_;
  }

  Assert(!HasInput("EndsTensorList"),
         "While slice/strided_slice has input EndsTensorList, requires "
         "opset_version >= 10");
  std::vector<int64_t> ends;
  if (HasInput("EndsTensor")) {
    auto info = GetInput("EndsTensor");
    Assert(TryGetInputValue("EndsTensor", &ends),
           "While slice/strided_slice has input EndsTensor, and it's not a "
           "constant tensor, then requires opset_version >= 10");
  } else {
    ends = ends_;
  }

  std::vector<int64_t> decrease_axis = DecreaseAxis();
  if (decrease_axis.empty()) {
    helper_->Slice(input_info[0].name, output_info[0].name, axes_, starts,
                   ends);
  } else {
    std::string node = helper_->Slice(input_info[0].name, axes_, starts, ends);
    helper_->Squeeze(node, output_info[0].name, decrease_axis);
  }
}

void SliceMapper::Opset10() {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");

  std::string starts = "";
  if (HasInput("StartsTensorList")) {
    auto info = GetInput("StartsTensorList");
    starts = helper_->ConcatIndices(info);
  } else if (HasInput("StartsTensor")) {
    auto info = GetInput("StartsTensor");
    starts = helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    starts = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, starts_);
  }

  std::string ends = "";
  if (HasInput("EndsTensorList")) {
    auto info = GetInput("EndsTensorList");
    ends = helper_->ConcatIndices(info);
  } else if (HasInput("EndsTensor")) {
    auto info = GetInput("EndsTensor");
    ends = helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    ends = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, ends_);
  }

  std::string strides = "";
  if (HasInput("StridesTensorList")) {
    auto info = GetInput("StridesTensorList");
    strides = helper_->ConcatIndices(info);
  } else if (HasInput("StridesTensor")) {
    auto info = GetInput("StridesTensor");
    strides =
        helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    if (strides_.size() == 0) {
      strides = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                  std::vector<int64_t>(axes_.size(), 1));
    } else {
      strides = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, strides_);
    }
  }

  auto axes = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, axes_);
  std::vector<int64_t> decrease_axis = DecreaseAxis();
  if (decrease_axis.empty()) {
    helper_->MakeNode("Slice",
                      {input_info[0].name, starts, ends, axes, strides},
                      {output_info[0].name});
  } else {
    auto out = helper_
                   ->MakeNode("Slice",
                              {input_info[0].name, starts, ends, axes, strides})
                   ->output(0);
    helper_->Squeeze(out, output_info[0].name, decrease_axis);
  }
}

}  // namespace paddle2onnx
