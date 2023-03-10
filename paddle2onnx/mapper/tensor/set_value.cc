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

#include "paddle2onnx/mapper/tensor/set_value.h"

namespace paddle2onnx {
REGISTER_MAPPER(set_value, SetValueMapper)

int32_t SetValueMapper::GetMinOpset(bool verbose) {
  if (none_axes_.size() > 0) {
    Error() << "Attribute none_axes is not supported." << std::endl;
    return -1;
  }
  if (axes_.size() > 1) {
    Error() << "Attribute axes is supported while it only contains 1 element."
            << std::endl;
    return -1;
  }
  if (steps_.size() > 1) {
    Error() << "ttribute steps is supported while it only contains 1 element."
            << std::endl;
    return -1;
  }
  if (GetInput("Input")[0].dtype == P2ODataType::BOOL) {
    Error() << "Input X with data type of boolean is not supported."
            << std::endl;
    return -1;
  }
  Logger(verbose, 12) << RequireOpset(12) << std::endl;
  return 12;
}

void SetValueMapper::Opset12() {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");
  std::string starts = "";
  if (HasInput("StartsTensorList")) {
    // if negtive value exists, not supported
    starts = helper_->ConcatIndices(GetInput("StartsTensorList"));
  } else {
    starts = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, starts_);
  }
  std::string ends = "";
  if (HasInput("EndsTensorList")) {
    ends = helper_->ConcatIndices(GetInput("EndsTensorList"));
  } else {
    // if out of range value in end exists, not supported
    ends = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, ends_);
  }

  auto input_tensor = input_info[0].name;
  std::string axes = helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64,
                                       int64_t(axes_[0]));
  // process out of range ends
  auto input_shape = helper_->MakeNode("Shape", {input_tensor})->output(0);
  auto gather_end_bound = helper_->MakeNode("Gather", {input_shape, axes});
  AddAttribute(gather_end_bound, "axis", int64_t(0));
  ends =
      helper_->MakeNode("Min", {gather_end_bound->output(0), ends})->output(0);

  std::string steps = "";
  if (HasInput("StepsTensorList")) {
    steps = helper_->ConcatIndices(GetInput("StepsTensorList"));
  } else {
    steps = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, steps_);
  }
  std::string value = "";
  int64_t value_rank = input_info[0].Rank();
  if (HasInput("ValueTensor")) {
    auto value_info = GetInput("ValueTensor");
    value = value_info[0].name;
    value_rank = value_info[0].Rank();
  } else {
    value_rank = shape_.size();
    int in_dtype = input_info[0].dtype;
    if (in_dtype == P2ODataType::INT32 || in_dtype == P2ODataType::INT64) {
      value = helper_->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                              int_values_);
    } else if (in_dtype == P2ODataType::FP32) {
      value = helper_->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                              fp32_values_);
    } else if (in_dtype == P2ODataType::FP64) {
      value = helper_->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                              fp64_values_);
    }
  }

  auto sliced_data =
      helper_->MakeNode("Slice", {input_tensor, starts, ends, axes, steps})
          ->output(0);

  auto sliced_shape = helper_->MakeNode("Shape", {sliced_data})->output(0);
  if (decrease_axes_.size() > 0 && value_rank != input_info[0].Rank()) {
    value = helper_->Unsqueeze(value, decrease_axes_);
  }
  auto expand_value =
      helper_->MakeNode("Expand", {value, sliced_shape})->output(0);

  auto indices = helper_
                     ->MakeNode("Range", {helper_->Squeeze(starts, {}),
                                          helper_->Squeeze(ends, {}),
                                          helper_->Squeeze(steps, {})})
                     ->output(0);
  if (axes_[0] == 0) {
    indices = helper_->Unsqueeze(indices, {1});
    helper_->MakeNode("ScatterND", {input_tensor, indices, expand_value},
                      {output_info[0].name});
  } else {
    std::vector<int64_t> indices_shape(input_info[0].Rank(), 1);
    indices_shape[axes_[0]] = -1;
    indices = helper_->Reshape(indices, indices_shape);
    auto one =
        helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
    if (axes_[0] == input_info[0].Rank() - 1) {
      auto part_shape = helper_->Slice(sliced_shape, {0}, {0}, {axes_[0]});
      auto tiled_shape = helper_->Concat({part_shape, one}, 0);
      indices = helper_->MakeNode("Tile", {indices, tiled_shape})->output(0);
    } else {
      auto part_0_shape = helper_->Slice(sliced_shape, {0}, {0}, {axes_[0]});
      auto part_1_shape = helper_->Slice(sliced_shape, {0}, {axes_[0] + 1},
                                         {input_info[0].Rank()});
      auto tiled_shape = helper_->Concat({part_0_shape, one, part_1_shape}, 0);
      indices = helper_->MakeNode("Tile", {indices, tiled_shape})->output(0);
    }
    auto scatter_node = helper_->MakeNode("ScatterElements",
                                          {input_tensor, indices, expand_value},
                                          {output_info[0].name});
    AddAttribute(scatter_node, "axis", axes_[0]);
  }
}

}  // namespace paddle2onnx
