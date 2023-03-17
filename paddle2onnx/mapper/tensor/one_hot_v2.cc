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

#include "paddle2onnx/mapper/tensor/one_hot_v2.h"

namespace paddle2onnx {
REGISTER_MAPPER(one_hot_v2, OneHotV2Mapper)

int32_t OneHotV2Mapper::GetMinOpset(bool verbose) {
  if (allow_out_of_range_) {
    Error() << "allow_out_of_range is not supported in one_hot_v2."
            << std::endl;
    return -1;
  }
  auto output_info = GetOutput("Out");
  if (output_info[0].dtype != dtype_) {
    Error() << "dtype attribute and output dtype do not match." << std::endl;
    return -1;
  }
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
}

void OneHotV2Mapper::Opset9() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string casted_input = helper_->AutoCast(
      input_info[0].name, input_info[0].dtype, P2ODataType::INT64);

  std::vector<int64_t> vals = {0, 1};
  std::string value_node =
      helper_->Constant(GetOnnxDtype(output_info[0].dtype), vals);

  std::string depth_node = "";
  if (HasInput("depth_tensor")) {
    auto input_depth_info = GetInput("depth_tensor");
    depth_node = input_depth_info[0].name;
  } else {
    depth_node =
        helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), depth_);
  }
  auto one_hot_node = helper_->MakeNode(
      "OneHot", {casted_input, depth_node, value_node}, {output_info[0].name});
}

}  // namespace paddle2onnx
