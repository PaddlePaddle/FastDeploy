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

#include "paddle2onnx/mapper/tensor/expand_v2.h"

namespace paddle2onnx {
REGISTER_MAPPER(expand_v2, ExpandV2Mapper)

void ExpandV2Mapper::Opset8() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  int dim_diff = 0;
  std::string shape = "";
  if (HasInput("Shape")) {
    auto shape_info = GetInput("Shape");
    dim_diff = shape_info[0].shape[0] - x_info[0].Rank();
    shape = helper_->AutoCast(shape_info[0].name, shape_info[0].dtype,
                              P2ODataType::INT64);
  } else if (HasInput("expand_shapes_tensor")) {
    auto shape_info = GetInput("expand_shapes_tensor");
    dim_diff = shape_info.size() - x_info[0].Rank();
    shape = helper_->ConcatIndices(shape_info);
  } else {
    std::vector<int64_t> shape_value;
    GetAttr("shape", &shape_value);
    dim_diff = shape_value.size() - x_info[0].Rank();
    shape = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, shape_value);
  }

  auto input_shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
  if (dim_diff > 0) {
    auto padding_shape = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                           std::vector<int64_t>(dim_diff, 1));
    input_shape = helper_->Concat({padding_shape, input_shape}, 0);
  }
  if (helper_->GetOpsetVersion() < 12) {
    // While opset < 12, Max cannot support int64 datatype with onnxruntime
    input_shape =
        helper_->AutoCast(input_shape, P2ODataType::INT64, P2ODataType::FP32);
    shape = helper_->AutoCast(shape, P2ODataType::INT64, P2ODataType::FP32);
    shape = helper_->MakeNode("Max", {input_shape, shape})->output(0);
    shape = helper_->AutoCast(shape, P2ODataType::FP32, P2ODataType::INT64);
  } else {
    shape = helper_->MakeNode("Max", {input_shape, shape})->output(0);
  }
  helper_->MakeNode("Expand", {x_info[0].name, shape}, {out_info[0].name});
}

}  // namespace paddle2onnx
