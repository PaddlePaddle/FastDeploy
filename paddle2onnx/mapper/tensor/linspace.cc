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

#include "paddle2onnx/mapper/tensor/linspace.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(linspace, LinspaceMapper)

int32_t LinspaceMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
};

void LinspaceMapper::Opset9() {
  auto start_info = GetInput("Start");
  auto stop_info = GetInput("Stop");
  auto num_info = GetInput("Num");
  auto output_info = GetOutput("Out");

  std::string cast_start = helper_->AutoCast(
      start_info[0].name, start_info[0].dtype, P2ODataType::FP32);
  std::string cast_stop = helper_->AutoCast(
      stop_info[0].name, stop_info[0].dtype, P2ODataType::FP32);

  auto sub_a_node = helper_->MakeNode("Sub", {cast_stop, cast_start});

  std::string one_node = helper_->Constant(GetOnnxDtype(num_info[0].dtype),
                                           std::vector<int32_t>(1, 1));

  auto sub_b_node = helper_->MakeNode("Sub", {num_info[0].name, one_node});

  std::string sub_b_float_node = helper_->AutoCast(
      sub_b_node->output(0), num_info[0].dtype, P2ODataType::FP32);

  auto step =
      helper_->MakeNode("Div", {sub_a_node->output(0), sub_b_float_node});

  std::string range_tensor = helper_->AutoCast(
      num_info[0].name, num_info[0].dtype, P2ODataType::INT64);

  std::string one_like_node = helper_->ConstOfShape(
      range_tensor, GetOnnxDtype(P2ODataType::FP32), static_cast<float>(1));

  auto none_zero_node = helper_->MakeNode("NonZero", {one_like_node});

  std::string trans_squeeze =
      helper_->Squeeze(none_zero_node->output(0), std::vector<int64_t>(1, 0));

  std::string cast_trans_squeeze =
      helper_->AutoCast(trans_squeeze, P2ODataType::INT64, P2ODataType::FP32);

  auto mul_node =
      helper_->MakeNode("Mul", {cast_trans_squeeze, step->output(0)});

  auto add_node = helper_->MakeNode("Add", {mul_node->output(0), cast_start});

  helper_->AutoCast(add_node->output(0), output_info[0].name, P2ODataType::FP32,
                    output_info[0].dtype);
}

}  // namespace paddle2onnx
