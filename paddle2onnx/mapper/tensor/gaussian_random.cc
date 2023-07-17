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

#include "paddle2onnx/mapper/tensor/gaussian_random.h"

namespace paddle2onnx {
REGISTER_MAPPER(gaussian_random, GaussianRandomMapper)

int32_t GaussianRandomMapper::GetMinOpset(bool verbose) {
  if (HasInput("ShapeTensor") && !IsConstantInput("ShapeTensor")) {
    Logger(verbose, 9)
        << "While ShapeTensor as input and it's not a constant tensor, "
        << RequireOpset(9) << std::endl;
    return 9;
  }
  if (HasInput("ShapeTensorList")) {
    Logger(verbose, 9) << "While ShapeTensorList as input, " << RequireOpset(9)
                       << std::endl;
    return 9;
  }
  return 7;
}

void GaussianRandomMapper::Opset7() {
  auto out_info = GetOutput("Out");
  std::string shape_tensor_name = "";
  std::vector<int64_t> shape;
  if (HasInput("ShapeTensor")) {
    if (!TryGetInputValue("ShapeTensor", &shape)) {
      auto shape_info = GetInput("ShapeTensor");
      shape_tensor_name = helper_->AutoCast(
          shape_info[0].name, shape_info[0].dtype, P2ODataType::INT64);
    }
  } else if (HasInput("ShapeTensorList")) {
    auto shape_info = GetInput("ShapeTensorList");
    shape_tensor_name = helper_->ConcatIndices(shape_info);
  } else {
    shape.assign(shape_.begin(), shape_.end());
  }
  if (out_info[0].Rank() == 0) {
    auto node = helper_->MakeNode("RandomNormal", {});
    AddAttribute(node, "dtype", GetOnnxDtype(out_info[0].dtype));
    AddAttribute(node, "mean", mean_);
    AddAttribute(node, "scale", std_);
    AddAttribute(node, "shape", std::vector<int64_t>(1, 1));
    AddAttribute(node, "seed", static_cast<float>(seed_));
    helper_->Squeeze(node->output(0), {out_info[0].name}, {0});
    return;
  }
  if (shape.size() > 0) {
    auto node = helper_->MakeNode("RandomNormal", {}, {out_info[0].name});
    AddAttribute(node, "dtype", GetOnnxDtype(out_info[0].dtype));
    AddAttribute(node, "mean", mean_);
    AddAttribute(node, "scale", std_);
    AddAttribute(node, "shape", shape_);
    AddAttribute(node, "seed", static_cast<float>(seed_));
  } else {
    auto tensor = helper_->ConstOfShape(
        shape_tensor_name, GetOnnxDtype(out_info[0].dtype), float(0));
    auto node =
        helper_->MakeNode("RandomNormalLike", {tensor}, {out_info[0].name});
    AddAttribute(node, "dtype", GetOnnxDtype(out_info[0].dtype));
    AddAttribute(node, "mean", mean_);
    AddAttribute(node, "scale", std_);
    AddAttribute(node, "seed", static_cast<float>(seed_));
  }
}

}  // namespace paddle2onnx
