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

#include "paddle2onnx/mapper/tensor/logical_op.h"

namespace paddle2onnx {
REGISTER_MAPPER(logical_and, LogicalOpMapper)
REGISTER_MAPPER(logical_or, LogicalOpMapper)
REGISTER_MAPPER(logical_xor, LogicalOpMapper)

void LogicalOpMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto out_info = GetOutput("Out");

  std::map<std::string, std::string> op_mapper;
  op_mapper["logical_and"] = "And";
  op_mapper["logical_or"] = "Or";
  op_mapper["logical_xor"] = "Xor";

  helper_->MakeNode(op_mapper[OpType()], {x_info[0].name, y_info[0].name},
                    {out_info[0].name});
}

}  // namespace paddle2onnx
