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

#include "paddle2onnx/mapper/tensor/stack.h"

namespace paddle2onnx {
REGISTER_MAPPER(stack, StackMapper)

void StackMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetOutput("Y");

  int32_t out_dtype = 0;
  std::vector<std::string> aligned_inputs =
      helper_->DtypeAlignment(x_info, &out_dtype);
  auto axis = axis_;
  if (axis < 0) {
    axis = axis + x_info[0].Rank() + 1;
  }
  for (size_t i = 0; i < aligned_inputs.size(); ++i) {
    aligned_inputs[i] =
        helper_->Unsqueeze(aligned_inputs[i], std::vector<int64_t>(1, axis));
  }
  auto out = helper_->Concat(aligned_inputs, axis_);
  helper_->AutoCast(out, y_info[0].name, out_dtype, y_info[0].dtype);
}

}  // namespace paddle2onnx
