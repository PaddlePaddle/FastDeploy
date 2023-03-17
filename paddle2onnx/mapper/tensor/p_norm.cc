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

#include "paddle2onnx/mapper/tensor/p_norm.h"

namespace paddle2onnx {
REGISTER_MAPPER(p_norm, PNormMapper)

void PNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string pnode =
      helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), porder_);

  auto abs_node = helper_->MakeNode("Abs", {input_info[0].name});
  auto pow_node = helper_->MakeNode("Pow", {abs_node->output(0), pnode});
  std::string reducesum_node = "";
  std::vector<int64_t> axes_val = {axis_};
  if (helper_->GetOpsetVersion() < 13) {
    auto node = helper_->MakeNode("ReduceSum", {pow_node->output(0)});
    AddAttribute(node, "axes", axes_val);
    AddAttribute(node, "keepdims", static_cast<int64_t>(keepdim_));
    reducesum_node = node->output(0);
  } else {
    std::string axes_node =
        helper_->Constant(GetOnnxDtype(P2ODataType::INT64), axes_val);
    auto node =
        helper_->MakeNode("ReduceSum", {pow_node->output(0), axes_node});
    AddAttribute(node, "keepdims", static_cast<int64_t>(keepdim_));
    reducesum_node = node->output(0);
  }

  std::string pnode1 =
      helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), 1.0 / porder_);
  helper_->MakeNode("Pow", {reducesum_node, pnode1}, {output_info[0].name});
}

}  // namespace paddle2onnx
