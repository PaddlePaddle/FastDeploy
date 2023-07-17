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

#include "paddle2onnx/mapper/tensor/tile.h"

namespace paddle2onnx {
REGISTER_MAPPER(tile, TileMapper)

void TileMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  bool has_repeats_tensor = HasInput("RepeatTimes");
  bool has_repeats_tensor_list = HasInput("repeat_times_tensor");
  std::string repeats = "";
  // NOTE(Aurelius84): we need to deprecate this branch in the future.
  if (has_repeats_tensor) {
    auto repeats_info = GetInput("RepeatTimes");
    repeats = helper_->AutoCast(repeats_info[0].name, repeats_info[0].dtype,
                                P2ODataType::INT64);
  } else if (has_repeats_tensor_list) {
    auto repeats_info = GetInput("repeat_times_tensor");
    repeats = helper_->ConcatIndices(repeats_info);
  } else if (IsAttrVar("repeat_times")) {
    auto repeats_info = GetAttrVar("repeat_times");
    if (repeats_info.size() == 1U) {  // repeat_times is a whole Tensor
      repeats = helper_->AutoCast(repeats_info[0].name, repeats_info[0].dtype,
                                  P2ODataType::INT64);
    } else {  // repeat_times is a tensor list
      repeats = helper_->ConcatIndices(repeats_info);
    }
  } else {
    std::vector<int64_t> values;
    GetAttr("repeat_times", &values);
    int64_t nums = values.size();
    for (int64_t i = 0; i < x_info[0].Rank() - nums; i++) {
      values.insert(values.begin(), 1);
    }
    repeats = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, values);
  }
  if (x_info[0].Rank() == 0) {
    auto unsqueeze = helper_->Unsqueeze(x_info[0].name, {0});
    helper_->MakeNode("Tile", {unsqueeze, repeats}, {out_info[0].name});
  } else {
    helper_->MakeNode("Tile", {x_info[0].name, repeats}, {out_info[0].name});
  }
}

}  // namespace paddle2onnx
