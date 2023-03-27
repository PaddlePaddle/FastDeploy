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

#include "paddle2onnx/mapper/tensor/index_sample.h"

namespace paddle2onnx {
REGISTER_MAPPER(index_sample, IndexSampleMapper)

int32_t IndexSampleMapper::GetMinOpset(bool verbose) {
  auto x_info = GetInput("X");
  auto index_info = GetInput("Index");
  if (x_info[0].Rank() != 2 || index_info[0].Rank() != 2) {
    Error() << "The rank of X and Index must be 2, but the rank of X is: "
            << x_info[0].Rank()
            << " , and the rank of Index is: " << index_info[0].Rank() << "."
            << std::endl;
    return -1;
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void IndexSampleMapper::Opset11() {
  auto x_info = GetInput("X");
  auto index_info = GetInput("Index");
  auto out_info = GetOutput("Out");
  auto node =
      helper_->MakeNode("GatherElements", {x_info[0].name, index_info[0].name},
                        {out_info[0].name});
  AddAttribute(node, "axis", static_cast<int64_t>(1));
}

}  // namespace paddle2onnx
