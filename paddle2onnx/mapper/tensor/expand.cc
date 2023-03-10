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

#include "paddle2onnx/mapper/tensor/expand.h"

namespace paddle2onnx {
REGISTER_MAPPER(expand, ExpandMapper)

void ExpandMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string expand_times = "";
  if (HasInput("expand_times_tensor")) {
    auto info = GetInput("expand_times_tensor");
    expand_times = helper_->ConcatIndices(info);
  } else if (HasInput("ExpandTimes")) {
    auto info = GetInput("ExpandTimes");
    expand_times = helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    expand_times = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, expand_times_);
  }

  helper_->MakeNode("Tile",
                      {input_info[0].name, expand_times},
                      {output_info[0].name});
}

}  // namespace paddle2onnx
