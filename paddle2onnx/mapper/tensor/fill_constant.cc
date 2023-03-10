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
#include "paddle2onnx/mapper/tensor/fill_constant.h"

#include <sstream>
#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(fill_constant, FillConstantMapper)

int32_t FillConstantMapper::GetMinOpset(bool verbose) {
  auto out_info = GetOutput("Out");
  auto onnx_dtype = GetOnnxDtype(out_info[0].dtype);
  if (onnx_dtype != ONNX_NAMESPACE::TensorProto::INT32 &&
      onnx_dtype != ONNX_NAMESPACE::TensorProto::INT64 &&
      onnx_dtype != ONNX_NAMESPACE::TensorProto::FLOAT &&
      onnx_dtype != ONNX_NAMESPACE::TensorProto::DOUBLE) {
    Error() << "Only support int32/int64/float32/float64 data type in "
               "fill_constant operator."
            << std::endl;
    return -1;
  }
  if (HasInput("ShapeTensorList")) {
    Logger(verbose, 9) << "While ShapeTensorList as input, " << RequireOpset(9) << std::endl;
    return 9;
  }
  if (HasInput("ShapeTensor") && !IsConstantInput("ShapeTensor")) {
    Logger(verbose, 9) << "While ShapeTensor as input and it's not a constant tensor, " << RequireOpset(9) << std::endl;
    return 9;
  }
  return 7;
}

float FillConstantMapper::GetFillValue() {
  float value = 0;
  if (str_value_.empty()) {
    value = value_;
  } else {
    if (str_value_ == "inf") {
      value = std::numeric_limits<float>::infinity();
    } else if (str_value_ == "-inf") {
      value = -std::numeric_limits<float>::infinity();
    } else if (str_value_ == "nan") {
      value = std::numeric_limits<float>::quiet_NaN();
    } else {
      std::stringstream convert_stream(str_value_);
      convert_stream >> value;
    }
  }
  if (HasInput("ValueTensor")) {
    value = 0.0;
  }
  return value;
}

void FillConstantMapper::Opset7() {
  auto out_info = GetOutput("Out");
  Assert(!HasInput("ShapeTensorList"), "While ShapeTensorList as input, requires opset_version>=9 for op fill_constant.");
  std::vector<int64_t> shape;
  if (HasInput("ShapeTensor")) {
    Assert(TryGetInputValue("ShapeTensor", &shape), "While ShapeTensor as input and it's not a constant tensor, requires opset_version>=9 for op fill_constant.");
  } else {
    GetAttr("shape", &shape);
  }
  float value = GetFillValue();
  if (HasInput("ValueTensor")) {
    auto value_info = GetInput("ValueTensor");
    auto value_tensor = helper_->AutoCast(value_info[0].name, value_info[0].dtype, out_info[0].dtype);
    auto out = helper_->Constant(shape, GetOnnxDtype(out_info[0].dtype), float(0.0));
    helper_->MakeNode("Add", {out, value_tensor}, {out_info[0].name});
  } else {
    helper_->Constant(out_info[0].name, shape, GetOnnxDtype(out_info[0].dtype), value);
  }
}

void FillConstantMapper::Opset9() {
  if (GetMinOpset() == 7) {
    return Opset7();
  }
  auto out_info = GetOutput("Out");
  bool shape_is_tensor = HasInput("ShapeTensor") || HasInput("ShapeTensorList");
  bool value_is_tensor = HasInput("ValueTensor");
  auto onnx_dtype = GetOnnxDtype(out_info[0].dtype);
  float value = GetFillValue();
  std::string out;
  if (shape_is_tensor) {
    std::string shape_name;
    if (HasInput("ShapeTensor")) {
      auto shape_info = GetInput("ShapeTensor");
      shape_name = helper_->AutoCast(shape_info[0].name, shape_info[0].dtype,
                                     P2ODataType::INT64);
    } else {
      auto shape_info = GetInput("ShapeTensorList");
      shape_name = helper_->ConcatIndices(shape_info);
    }

    auto node = helper_->MakeNode("ConstantOfShape", {shape_name});
    auto attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
    auto tensor = attr->mutable_t();
    tensor->set_name(out_info[0].name);
    tensor->set_data_type(onnx_dtype);
    tensor->add_dims(1);
    if (onnx_dtype == ONNX_NAMESPACE::TensorProto::INT32) {
      std::vector<int32_t> data(1);
      data[0] = static_cast<int32_t>(value);
      auto ptr = reinterpret_cast<char*>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(int32_t)));
    } else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::INT64) {
      std::vector<int64_t> data(1);
      data[0] = static_cast<int64_t>(value);
      auto ptr = reinterpret_cast<char*>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(int64_t)));
    } else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
      std::vector<float> data(1, value_);
      auto ptr = reinterpret_cast<char*>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(float)));
    } else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
      std::vector<double> data(1);
      data[0] = static_cast<double>(value);
      auto ptr = reinterpret_cast<char*>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(double)));
    }
    out = node->output(0);
  } else {
    std::vector<int64_t> shape;
    GetAttr("shape", &shape);
    out = helper_->Constant(shape, onnx_dtype, value);
  }
  if (value_is_tensor) {
    auto value_info = GetInput("ValueTensor");
    std::string cast_value = helper_->AutoCast(
        value_info[0].name, value_info[0].dtype, out_info[0].dtype);
    helper_->MakeNode("Add", {out, cast_value}, {out_info[0].name});
  } else {
    helper_->MakeNode("Identity", {out}, {out_info[0].name});
  }
}

}  // namespace paddle2onnx
