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

#include "paddle2onnx/mapper/tensor/expand_as.h"

#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(expand_as_v2, ExpandAsMapper)

int32_t ExpandAsMapper::GetMinOpset(bool verbose) {
  if (target_shape_.size() == 0 && !HasInput("target_tensor")) {
    Error() << "Attribute `target_shape` or input tensor `target_tensor` is "
               "not exist"
            << std::endl;
    return -1;
  }
  Logger(verbose, 8) << RequireOpset(8) << std::endl;
  return 8;
};

void ExpandAsMapper::Opset8() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string target_shape = "";
  if (HasInput("target_tensor")) {
    auto info = GetInput("target_tensor");
    target_shape = helper_->MakeNode("Shape", {info[0].name})->output(0);
  } else {
    target_shape =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, target_shape_);
  }

  helper_->MakeNode("Expand", {input_info[0].name, target_shape},
                    {output_info[0].name});
}

}  // namespace paddle2onnx
