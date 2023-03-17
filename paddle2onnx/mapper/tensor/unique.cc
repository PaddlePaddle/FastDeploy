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

#include "paddle2onnx/mapper/tensor/unique.h"

namespace paddle2onnx {
REGISTER_MAPPER(unique, UniqueMapper)

int32_t UniqueMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void UniqueMapper::Opset11() {
  auto intput_info = GetInput("X");
  auto output_out_info = GetOutput("Out");
  auto output_indices_info = GetOutput("Indices");
  auto output_index_info = GetOutput("Index");
  auto output_counts_info = GetOutput("Counts");

  std::string out_index_name = "out_index";
  std::string out_indices_name = "out_indices";
  std::string out_counts_name = "out_counts";
  auto node = helper_->MakeNode("Unique", {intput_info[0].name},
                                {output_out_info[0].name, out_indices_name,
                                 out_index_name, out_counts_name});
  if (axis_.size()) {
    AddAttribute(node, "axis", axis_[0]);
  }
  helper_->AutoCast(out_index_name, output_index_info[0].name,
                    P2ODataType::INT64, output_index_info[0].dtype);
  helper_->AutoCast(out_indices_name, output_indices_info[0].name,
                    P2ODataType::INT64, output_indices_info[0].dtype);
  helper_->AutoCast(out_counts_name, output_counts_info[0].name,
                    P2ODataType::INT64, output_counts_info[0].dtype);
}

}  // namespace paddle2onnx
