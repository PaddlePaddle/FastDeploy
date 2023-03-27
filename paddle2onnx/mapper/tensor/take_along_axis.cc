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

#include "paddle2onnx/mapper/tensor/take_along_axis.h"

namespace paddle2onnx {
REGISTER_MAPPER(take_along_axis, TakeAlongAxisMapper)

int32_t TakeAlongAxisMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void TakeAlongAxisMapper::Opset11() {
  auto x_info = GetInput("Input");
  auto index_info = GetInput("Index");
  auto out_info = GetOutput("Result");

  auto axis = axis_;

  auto node =
      helper_->MakeNode("GatherElements", {x_info[0].name, index_info[0].name},
                        {out_info[0].name});
  AddAttribute(node, "axis", axis);
}

}  // namespace paddle2onnx
