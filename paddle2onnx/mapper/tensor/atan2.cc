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

#include "paddle2onnx/mapper/tensor/atan2.h"
#define M_PI 3.14159265358979323846 /* pi */

namespace paddle2onnx {
REGISTER_MAPPER(atan2, Atan2Mapper)

int32_t Atan2Mapper::GetMinOpset(bool verbose) {
  if (GetInput("X1")[0].dtype == P2ODataType::INT32 ||
      GetInput("X2")[0].dtype == P2ODataType::INT32 ||
      GetInput("X1")[0].dtype == P2ODataType::INT64 ||
      GetInput("X2")[0].dtype == P2ODataType::INT64) {
    Error() << "The input dtype should be float32 or float64. " << std::endl;
    return -1;
  }
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
}

void Atan2Mapper::Opset9() {
  auto x_info = GetInput("X1");
  auto y_info = GetInput("X2");
  auto out_info = GetOutput("Out");

  std::string input_x_name = x_info[0].name;
  std::string input_y_name = y_info[0].name;
  auto dtype = P2ODataType::FP32;
  if (x_info[0].dtype == P2ODataType::FP64 ||
      y_info[0].dtype == P2ODataType::FP64) {
    input_x_name =
        helper_->AutoCast(x_info[0].name, x_info[0].dtype, P2ODataType::FP32);
    input_y_name =
        helper_->AutoCast(y_info[0].name, y_info[0].dtype, P2ODataType::FP32);
  }
  auto div = helper_->MakeNode("Div", {input_x_name, input_y_name});
  auto atan = helper_->MakeNode("Atan", {div->output(0)});

  std::string zero_node =
      helper_->Constant(GetOnnxDtype(dtype), std::vector<float>{0.0});

  auto minus_node = helper_->MakeNode("Less", {input_y_name, zero_node});

  std::string condition_node =
      helper_->AutoCast(minus_node->output(0), dtype, P2ODataType::BOOL);

  std::string pi_node =
      helper_->Constant(GetOnnxDtype(dtype), std::vector<float>{static_cast<float>(M_PI)});

  auto sign_node = helper_->MakeNode("Sign", {input_x_name});

  auto mul_node = helper_->MakeNode("Mul", {sign_node->output(0), pi_node});

  auto where_node = helper_->MakeNode(
      "Where", {condition_node, mul_node->output(0), zero_node});

  auto add_node =
      helper_->MakeNode("Add", {atan->output(0), where_node->output(0)});

  helper_->AutoCast(add_node->output(0), out_info[0].name, dtype,
                    out_info[0].dtype);
}

}  // namespace paddle2onnx
