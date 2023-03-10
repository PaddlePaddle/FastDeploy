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

#include "paddle2onnx/mapper/tensor/not_equal.h"

namespace paddle2onnx {
REGISTER_MAPPER(not_equal, NotEqualMapper)

int32_t NotEqualMapper::GetMinOpset(bool verbose) {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  if (x_info[0].dtype == P2ODataType::FP32 ||
      x_info[0].dtype == P2ODataType::FP64) {
    Logger(verbose, 11) << "While input is dtype of float32/float64, "
                        << RequireOpset(11) << std::endl;
    return 11;
  }
  return 7;
}

void NotEqualMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto out_info = GetOutput("Out");

  int out_dtype = 0;
  std::vector<std::string> aligned_inputs =
      helper_->DtypeAlignment({x_info[0], y_info[0]}, &out_dtype);

  auto output = helper_->MakeNode("Equal", aligned_inputs)->output(0);
  helper_->MakeNode("Not", {output}, {out_info[0].name});
}

}  // namespace paddle2onnx
