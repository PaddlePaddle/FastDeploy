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
#include "paddle2onnx/mapper/tensor/fill_like.h"

namespace paddle2onnx {

REGISTER_MAPPER(fill_any_like, FillLikeMapper)
REGISTER_MAPPER(fill_zeros_like, FillLikeMapper)

void FillLikeMapper::Opset9() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  bool is_fixed_shape = true;
  for (size_t i = 0; i < input_info[0].shape.size(); ++i) {
    if (input_info[0].shape[i] < 0) {
      is_fixed_shape = false;
    }
  }
  if (is_fixed_shape) {
    helper_->Constant(output_info[0].name, input_info[0].shape,
                      GetOnnxDtype(output_info[0].dtype), value_);
    return;
  }
  auto shape_node = helper_->MakeNode("Shape", {input_info[0].name});
  int64_t dtype = output_info[0].dtype;
  // There's some problem with tensorrt with `ConstantOfShape`
  // Maybe we should use a graph pass to solve this problem
  // but now we just to avoid using `ConstantOfShape`
  auto const_node = helper_->Constant({1}, GetOnnxDtype(dtype), value_);
  helper_->MakeNode("Expand", {const_node, shape_node->output(0)},
                    {output_info[0].name});
  //    helper_->ConstOfShape(shape_node->output(0), output_info[0].name,
  //                         GetOnnxDtype(dtype), value_);
}

}  // namespace paddle2onnx
