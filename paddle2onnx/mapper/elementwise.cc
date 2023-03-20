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
#include "paddle2onnx/mapper/elementwise.h"

namespace paddle2onnx {

REGISTER_MAPPER(elementwise_add, ElementwiseMapper)
REGISTER_MAPPER(elementwise_sub, ElementwiseMapper)
REGISTER_MAPPER(elementwise_div, ElementwiseMapper)
REGISTER_MAPPER(elementwise_mul, ElementwiseMapper)
REGISTER_MAPPER(elementwise_min, ElementwiseMapper)
REGISTER_MAPPER(elementwise_max, ElementwiseMapper)
REGISTER_MAPPER(elementwise_pow, ElementwiseMapper)
REGISTER_MAPPER(elementwise_mod, ElementWiseModMapper)
REGISTER_MAPPER(elementwise_floordiv, ElementWiseFloordivMapper)

int32_t ElementwiseMapper::GetMinOpset(bool verbose) {
  if (OpType() == "elementwise_min" || OpType() == "elementwise_max") {
    Logger(verbose, 8) << RequireOpset(8) << std::endl;
    return 8;
  }
  return 7;
}

void ElementwiseMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");
  auto iter = op_mapper_.find(OpType());
  Assert(op_mapper_.end() != iter,
         "Cannot find " + OpType() + " in elementwise op_mapper.");

  auto x_name = input_x_info[0].name;
  auto y_name = input_y_info[0].name;
  if (input_x_info[0].dtype == P2ODataType::BOOL &&
      input_y_info[0].dtype == P2ODataType::BOOL) {
    x_name =
        helper_->AutoCast(x_name, input_x_info[0].dtype, P2ODataType::INT32);
    y_name =
        helper_->AutoCast(y_name, input_y_info[0].dtype, P2ODataType::INT32);
  }

  std::string output_name;
  if (axis_ == -1 || axis_ == (input_x_info[0].Rank() - 1) ||
      input_x_info[0].Rank() == input_y_info[0].Rank()) {
    output_name = helper_->MakeNode(iter->second, {x_name, y_name})->output(0);
  } else {
    std::vector<int64_t> broadcast_shape(input_x_info[0].Rank(), 1);
    for (int i = axis_; i < axis_ + input_y_info[0].Rank(); ++i) {
      broadcast_shape[i] = input_y_info[0].shape[i - axis_];
    }
    std::string broadcast_shape_node =
        helper_->Constant(GetOnnxDtype(P2ODataType::INT64), broadcast_shape);
    auto y_node = helper_->MakeNode("Reshape", {y_name, broadcast_shape_node});
    output_name =
        helper_->MakeNode(iter->second, {x_name, y_node->output(0)})->output(0);
  }

  if (input_x_info[0].dtype == P2ODataType::BOOL &&
      input_y_info[0].dtype == P2ODataType::BOOL) {
    helper_->AutoCast(output_name, output_info[0].name, P2ODataType::INT32,
                      P2ODataType::BOOL);
  } else {
    helper_->MakeNode("Identity", {output_name}, {output_info[0].name});
  }
}

void ElementWiseModMapper::Opset10() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");
  int64_t fmod = 0;
  if (input_y_info[0].dtype == P2ODataType::INT32 ||
      input_y_info[0].dtype == P2ODataType::INT64) {
    if (this->deploy_backend == "tensorrt") {
      auto x = helper_->AutoCast(input_x_info[0].name, input_x_info[0].dtype,
                                 input_y_info[0].dtype);
      auto times =
          helper_->MakeNode("Div", {input_x_info[0].name, input_y_info[0].name})
              ->output(0);
      auto result =
          helper_->MakeNode("Mul", {input_y_info[0].name, times})->output(0);
      helper_->MakeNode("Sub", {input_x_info[0].name, result},
                        {output_info[0].name});
      return;
    }
    auto mod_node =
        helper_->MakeNode("Mod", {input_x_info[0].name, input_y_info[0].name},
                          {output_info[0].name});
    AddAttribute(mod_node, "fmod", fmod);
    return;
  }

  fmod = 1;

  auto abs_x_node = helper_->MakeNode("Abs", {input_x_info[0].name});
  auto abs_y_node = helper_->MakeNode("Abs", {input_y_info[0].name});

  auto dtype = input_y_info[0].dtype;
  std::vector<float> val_0 = {0.0};

  std::string zero_node = helper_->Constant(GetOnnxDtype(dtype), val_0);

  auto mod_node =
      helper_->MakeNode("Mod", {abs_x_node->output(0), abs_y_node->output(0)});
  AddAttribute(mod_node, "fmod", fmod);

  auto neg_node = helper_->MakeNode("Neg", {mod_node->output(0)});

  auto less_node = helper_->MakeNode("Less", {input_x_info[0].name, zero_node});

  std::string condition_node =
      helper_->AutoCast(less_node->output(0), dtype, P2ODataType::BOOL);

  auto mod_res_node = helper_->MakeNode(
      "Where", {condition_node, neg_node->output(0), mod_node->output(0)});

  auto mod_y_add_node =
      helper_->MakeNode("Add", {mod_res_node->output(0), input_y_info[0].name});

  auto mod_y_mul_node =
      helper_->MakeNode("Mul", {mod_res_node->output(0), input_y_info[0].name});

  auto mod_y_mul_less_node =
      helper_->MakeNode("Less", {mod_y_mul_node->output(0), zero_node});

  std::string mod_y_mul_condition_node = helper_->AutoCast(
      mod_y_mul_less_node->output(0), dtype, P2ODataType::BOOL);

  helper_->MakeNode("Where",
                    {mod_y_mul_condition_node, mod_y_add_node->output(0),
                     mod_res_node->output(0)},
                    {output_info[0].name});
}

void ElementWiseFloordivMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  bool is_int = false;
  if (input_x_info[0].dtype <= 3 || input_x_info[0].dtype == 20 ||
      input_y_info[0].dtype <= 3 || input_y_info[0].dtype == 20) {
    is_int = true;
  }
  if (axis_ == -1 || axis_ == input_x_info[0].Rank() - 1 ||
      input_x_info[0].Rank() == input_y_info[0].Rank()) {
    if (is_int) {
      helper_->MakeNode("Div", {input_x_info[0].name, input_y_info[0].name},
                        {output_info[0].name});
    } else {
      auto div_node = helper_->MakeNode(
          "Div", {input_x_info[0].name, input_y_info[0].name});
      helper_->MakeNode("Floor", {div_node->output(0)}, {output_info[0].name});
    }
  } else {
    std::vector<int64_t> broadcast_shape;
    broadcast_shape.resize(axis_ + input_x_info[0].Rank(), 1);
    for (auto i = 0; i < input_y_info[0].Rank(); ++i) {
      broadcast_shape[axis_ + i] = input_y_info[0].shape[i];
    }
    std::string broadcast_shape_node =
        helper_->Constant(GetOnnxDtype(P2ODataType::INT64), broadcast_shape);
    auto y_node = helper_->MakeNode(
        "Reshape", {input_y_info[0].name, broadcast_shape_node});
    if (is_int) {
      helper_->MakeNode("Div", {input_x_info[0].name, y_node->output(0)},
                        {output_info[0].name});
    } else {
      auto div_node =
          helper_->MakeNode("Div", {input_x_info[0].name, y_node->output(0)});
      helper_->MakeNode("Floor", {div_node->output(0)}, {output_info[0].name});
    }
  }
}

}  // namespace paddle2onnx
