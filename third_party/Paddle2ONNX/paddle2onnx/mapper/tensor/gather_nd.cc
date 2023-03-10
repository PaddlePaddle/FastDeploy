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

#include "paddle2onnx/mapper/tensor/gather_nd.h"

namespace paddle2onnx {
REGISTER_MAPPER(gather_nd, GatherNdMapper)

int32_t GatherNdMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void GatherNdMapper::Opset11() {
  auto input_x_info = GetInput("X");
  auto input_index_info = GetInput("Index");
  auto output_info = GetOutput("Out");

  std::string index_node = helper_->AutoCast(
      input_index_info[0].name, input_index_info[0].dtype, P2ODataType::INT64);

  helper_->MakeNode("GatherND", {input_x_info[0].name, index_node},
                    {output_info[0].name});
}

}  // namespace paddle2onnx
