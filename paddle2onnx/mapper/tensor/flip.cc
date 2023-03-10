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

#include "paddle2onnx/mapper/tensor/flip.h"

namespace paddle2onnx {
REGISTER_MAPPER(flip, FlipMapper)

int32_t FlipMapper::GetMinOpset(bool verbose) {
  auto input_info = parser_->GetOpInput(block_idx_, op_idx_, "X");
  for (auto i = 0; i < axes_.size(); i++) {
    if (input_info[0].shape[axes_[i]] <= 0) {
      Error() << "The dimension in axis of input must be fixed for flip "
                 "operator, but now the input shape in axis is unkown."
              << std::endl;
      return -1;
    }
  }
  return 7;
}

void FlipMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string input_name = input_info[0].name;
  bool need_convert = false;
  if (input_info[0].dtype == P2ODataType::BOOL ||
      input_info[0].dtype == P2ODataType::FP64) {
    need_convert = true;
    input_name = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
  }

  std::string temp_input = input_name;
  for (auto i = 0; i < axes_.size(); ++i) {
    int64_t axis = axes_[i];
    if (input_info[0].shape[axis] == 1) {
      if (i != axes_.size() - 1) {
        continue;
      }
      if (need_convert) {
        input_name = helper_->AutoCast(temp_input, output_info[0].name,
                                       P2ODataType::FP32, output_info[0].dtype);
      } else {
        auto out_node =
            helper_->MakeNode("Identity", {temp_input}, {output_info[0].name});
      }
    } else {
      std::vector<int64_t> split;
      split.resize(input_info[0].shape[axis], 1);
      std::vector<std::string> splits_outputs =
          helper_->Split(temp_input, split, axis);
      std::vector<std::string> reversed_splits;
      for (int64_t index = splits_outputs.size() - 1; index >= 0; --index) {
        reversed_splits.push_back(splits_outputs[index]);
      }
      if (i != axes_.size() - 1) {
        auto concat_node = helper_->MakeNode("Concat", reversed_splits);
        AddAttribute(concat_node, "axis", axis);
        temp_input = concat_node->output(0);
      } else {
        if (need_convert) {
          auto concat_node = helper_->MakeNode("Concat", reversed_splits);
          AddAttribute(concat_node, "axis", axis);
          helper_->AutoCast(concat_node->output(0), output_info[0].name,
                            P2ODataType::FP32, output_info[0].dtype);
        } else {
          auto concat_node = helper_->MakeNode("Concat", reversed_splits,
                                               {output_info[0].name});
          AddAttribute(concat_node, "axis", axis);
        }
      }
    }
  }
}

}  // namespace paddle2onnx
