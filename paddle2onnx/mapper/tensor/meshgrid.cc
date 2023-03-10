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

#include "paddle2onnx/mapper/tensor/meshgrid.h"

namespace paddle2onnx {
REGISTER_MAPPER(meshgrid, MeshgridMapper)

void MeshgridMapper::Opset8() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  std::vector<std::string> x_shapes(x_info.size());
  for (size_t i = 0; i < x_info.size(); ++i) {
    x_shapes[i] = helper_->MakeNode("Shape", {x_info[i].name})->output(0);
  }
  auto out_shape = helper_->Concat(x_shapes, 0);
  for (size_t i = 0; i < x_info.size(); ++i) {
    std::vector<std::string> intermediate_shape(x_info.size());
    for (size_t j = 0; j < x_info.size(); ++j) {
      if (j == i) {
        intermediate_shape[j] = x_shapes[i];
      } else {
        intermediate_shape[j] = helper_->Constant(
            ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>(1, 1));
      }
    }
    auto t_reshaped = helper_->Concat(intermediate_shape, 0);
    t_reshaped =
        helper_->MakeNode("Reshape", {x_info[i].name, t_reshaped})->output(0);
    helper_->MakeNode("Expand", {t_reshaped, out_shape}, {out_info[i].name});
  }
}

}  // namespace paddle2onnx
