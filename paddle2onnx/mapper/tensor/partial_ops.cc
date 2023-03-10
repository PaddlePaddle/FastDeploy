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

#include "paddle2onnx/mapper/tensor/partial_ops.h"

namespace paddle2onnx {
REGISTER_MAPPER(partial_sum, PartialOpsMapper)
REGISTER_MAPPER(partial_concat, PartialOpsMapper)

int32_t PartialOpsMapper::GetMinOpset(bool verbose) {
  auto input_info = GetInput("X");
  for (auto &in : input_info) {
    if (in.Rank() != 2) {
      Error() << "The input dim of partial_sum OP must be 2." << std::endl;
      return -1;
    }
  }
  if (start_index_ < 0) {
    start_index_ = start_index_ + input_info[0].shape[1];
  }
  int64_t batch_size = input_info[0].shape[0];
  int64_t max_length = input_info[0].shape[1];
  for (auto &in : input_info) {
    if (in.shape[0] != batch_size || in.shape[1] != max_length) {
      Error()
          << "The batch_size and max_length of all inputs must be same in " +
                 OpType() + " OP."
          << std::endl;
      return -1;
    }
  }
  if (max_length < start_index_) {
    Error() << "start_index must be less than input len in " + OpType() + " OP."
            << std::endl;
    return -1;
  }
  if (length_ > 0 && start_index_ + length_ > max_length) {
    Error() << "start_index + length is larger than input length in " +
                   OpType() + " OP."
            << std::endl;
    return -1;
  }
  auto iter = op_mapper_.find(OpType());
  if (op_mapper_.end() == iter) {
    Error() << "Cannot find " + OpType() + " in partial op_mapper."
            << std::endl;
    return -1;
  }
  return 7;
}

void PartialOpsMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  int64_t end;
  if (length_ < 0) {
    end = input_info[0].shape[1];
  } else {
    end = start_index_ + length_;
  }
  std::vector<std::string> slice_outputs;
  for (auto &in : input_info) {
    auto out = helper_->Slice(in.name, {1}, {start_index_}, {end});
    std::string casted_node =
        helper_->AutoCast(out, in.dtype, P2ODataType::FP32);
    slice_outputs.push_back(casted_node);
  }
  auto iter = op_mapper_.find(OpType());
  auto node = helper_->MakeNode(iter->second, slice_outputs);
  if (iter->second == "Concat") {
    AddAttribute(node, "axis", static_cast<int64_t>(1));
  }
  helper_->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                    output_info[0].dtype);
}

}  // namespace paddle2onnx
