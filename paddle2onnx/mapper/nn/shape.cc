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

#include "paddle2onnx/mapper/nn/shape.h"

namespace paddle2onnx {
REGISTER_MAPPER(shape, ShapeMapper)

void ShapeMapper::Opset7() {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");

  auto shape_out = helper_->MakeNode("Shape", {input_info[0].name})->output(0);
  helper_->AutoCast(shape_out, output_info[0].name, P2ODataType::INT64,
                    output_info[0].dtype);
}

}  // namespace paddle2onnx
