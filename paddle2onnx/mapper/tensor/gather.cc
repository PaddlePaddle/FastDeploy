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

#include "paddle2onnx/mapper/tensor/gather.h"

namespace paddle2onnx {
REGISTER_MAPPER(gather, GatherMapper)

int32_t GatherMapper::GetMinOpset(bool verbose) {
  if (HasInput("Axis")) {
    if (!IsConstantInput("Axis")) {
      Error() << "Parameter axis as input tensor is not supported."
              << std::endl;
      return -1;
    }
  }
  auto index_info = GetInput("Index");
  if (index_info[0].shape.size() > 1) {
    Logger(verbose, 11) << "While rank of index > 1, " << RequireOpset(11)
                        << std::endl;
    return 11;
  }
  return 7;
}

void GatherMapper::Opset7() {
  auto x_info = GetInput("X");
  auto index_info = GetInput("Index");
  auto out_info = GetOutput("Out");

  bool has_input_axis = HasInput("Axis");
  auto axis = axis_;
  if (has_input_axis) {
    std::vector<int64_t> axes;
    Assert(TryGetInputValue("Axis", &axes),
           "Paddle2ONNX does not support axis as input tensor for operator: "
           "gather.");
    axis = axes[0];
  }
  Assert(index_info[0].shape.size() == 1,
         "Paddle2ONNX: While rank of index > 1, opset must >= 11 for operator: "
         "gather.");
  auto node = helper_->MakeNode("Gather", {x_info[0].name, index_info[0].name},
                                {out_info[0].name});
  AddAttribute(node, "axis", axis);
}

void GatherMapper::Opset11() {
  auto x_info = GetInput("X");
  auto index_info = GetInput("Index");
  auto out_info = GetOutput("Out");

  bool has_input_axis = HasInput("Axis");
  auto axis = axis_;
  if (has_input_axis) {
    std::vector<int64_t> axes;
    Assert(TryGetInputValue("Axis", &axes),
           "Paddle2ONNX does not support axis as input tensor for operator: "
           "gather.");
    axis = axes[0];
  }
  if (index_info[0].shape.size() == 1) {
    auto node = helper_->MakeNode(
        "Gather", {x_info[0].name, index_info[0].name}, {out_info[0].name});
    AddAttribute(node, "axis", axis);
  } else {
    auto index = helper_->AutoCast(index_info[0].name, index_info[0].dtype,
                                   P2ODataType::INT64);
    helper_->MakeNode("GatherND", {x_info[0].name, index_info[0].name},
                      {out_info[0].name});
  }
}

}  // namespace paddle2onnx
