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

#pragma once

#include <onnx/onnx_pb.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle2onnx/mapper/register_mapper.h"
#include "paddle2onnx/parser/parser.h"

namespace paddle2onnx {

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const int64_t& value);
void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const float& value);
void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::string& value);
void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::vector<int64_t>& values);
void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::vector<float>& values);
void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name,
                  ONNX_NAMESPACE::TensorProto_DataType dtype);

ONNX_NAMESPACE::TensorProto_DataType GetOnnxDtype(int32_t paddle_dtype);
std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeConstant(const std::string& name,
                                                        const Weight& weight);
std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto> MakeValueInfo(
    const TensorInfo& info);

struct QuantizeInfo {
 public:
  std::vector<float> scale_;
  std::vector<int64_t> zeros_;
  std::string zeros_node_;
  std::string scale_node_;
  int64_t quantize_axis_;

  QuantizeInfo() {}
  QuantizeInfo(const std::vector<float>& scale,
               const std::vector<int64_t>& zeros, const std::string& scale_node,
               const std::string& zeros_node, const int64_t& quantize_axis) {
    zeros_node_ = zeros_node;
    scale_node_ = scale_node;
    quantize_axis_ = quantize_axis;
    scale_.resize(scale.size());
    memcpy(scale_.data(), scale.data(), scale.size() * sizeof(float));
    zeros_.resize(zeros.size());
    memcpy(zeros_.data(), zeros.data(), zeros.size() * sizeof(int64_t));
  }
};

class OnnxHelper {
 public:
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> nodes;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> value_infos;
  int32_t opset_version = 7;
  // Use updated_params to store params that were changed during conversion
  std::map<std::string, Weight> updated_params;
  // Use quantize_info to record quantization-related information, scale and
  // zero information corresponding to each tensor
  std::map<std::string, QuantizeInfo> quantize_info;

  void Clear() { nodes.clear(); }

  void SetOpsetVersion(int32_t op_v) { opset_version = op_v; }

  int32_t GetOpsetVersion() { return opset_version; }

  template <typename T>
  bool TryGetTensorValue(const std::string& name, std::vector<T>* value);

  std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto> MakeValueInfo(
      const std::string& name, const int32_t& dtype,
      std::vector<int64_t>& shape);

  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeNode(
      const std::string& op_type, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs);
  // we use this function to generate some temporary node
  // we do not need to define the outputs, because the outputs
  // is generate by MapperHelper, which will make sure there's no
  // name confict problem
  // the parameter `num_outputs` will define the number of output names
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeNode(
      const std::string& op_type, const std::vector<std::string>& inputs,
      int num_outputs = 1);

  template <typename T>
  std::string ConstOfShape(const std::string& input, const std::string& output,
                           ONNX_NAMESPACE::TensorProto_DataType dtype, T value);
  template <typename T>
  std::string ConstOfShape(const std::string& input,
                           ONNX_NAMESPACE::TensorProto_DataType dtype, T value);

  std::string AutoCast(const std::string& input, int32_t input_paddle_dtype,
                       int32_t to_paddle_dtype);
  std::string AutoCast(const std::string& input, const std::string& output,
                       int32_t input_paddle_dtype, int32_t to_paddle_dtype);

  // Helper function for PaddlePaddle's shape tensor list inputs
  // will cast all data type to int64
  // will make sure all inputs to be 1-D tensor
  // will concat them as output
  std::string ConcatIndices(const std::vector<TensorInfo>& indices);
  std::vector<std::string> DtypeAlignment(
      const std::vector<TensorInfo>& input_info, int32_t* out_dtype);
  std::string Clip(const std::string& input, const float& min, const float& max,
                   const int32_t& in_dtype);
  std::string Clip(const std::string& input, const std::string& output,
                   const float& min, const float& max, const int32_t& in_dtype);
  std::string Squeeze(const std::string& input,
                      const std::vector<int64_t>& axes);
  std::string Squeeze(const std::string& input, const std::string& output,
                      const std::vector<int64_t>& axes);
  std::string Unsqueeze(const std::string& input,
                        const std::vector<int64_t>& axes);
  std::string Unsqueeze(const std::string& input, const std::string& output,
                        const std::vector<int64_t>& axes);
  std::string Reshape(const std::string& input, const std::string& output,
                      const std::vector<int64_t>& shape);
  std::string Reshape(const std::string& input,
                      const std::vector<int64_t>& shape);
  std::string Flatten(const std::string& input, const std::string& output);
  std::string Flatten(const std::string& input);
  std::string Slice(const std::string& input, const std::string& output,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends);
  std::string Slice(const std::string& input, const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends);
  std::string Concat(const std::vector<std::string>& input,
                     const std::string& output, int64_t axis);
  std::string Concat(const std::vector<std::string>& input, int64_t axis);

  std::string Transpose(const std::string& input, const std::string& output,
                        const std::vector<int64_t>& perm);
  std::string Transpose(const std::string& input,
                        const std::vector<int64_t>& perm);

  std::vector<std::string> Split(const std::string& input,
                                 const std::vector<std::string>& outputs,
                                 const std::vector<int64_t>& split,
                                 int64_t axis);
  std::vector<std::string> Split(const std::string& input,
                                 const std::vector<int64_t>& split,
                                 int64_t axis);

  template <typename T>
  std::string Constant(const std::string& output,
                       ONNX_NAMESPACE::TensorProto_DataType dtype,
                       const std::vector<T>& value);
  template <typename T>
  std::string Constant(ONNX_NAMESPACE::TensorProto_DataType dtype,
                       const std::vector<T>& value);
  template <typename T>
  std::string Constant(const std::string& output,
                       const std::vector<int64_t>& shape,
                       ONNX_NAMESPACE::TensorProto_DataType dtype, T value);
  template <typename T>
  std::string Constant(const std::vector<int64_t>& shape,
                       ONNX_NAMESPACE::TensorProto_DataType dtype, T value);

  template <typename T>
  std::string Constant(const std::vector<int64_t>& shape,
                       ONNX_NAMESPACE::TensorProto_DataType dtype,
                       std::vector<T>& value);

  template <typename T>
  std::string Assign(const std::string& output,
                     const ONNX_NAMESPACE::TensorProto_DataType& dtype,
                     const std::vector<int64_t>& shape,
                     const std::vector<T>& value);
  template <typename T>
  std::string Assign(const ONNX_NAMESPACE::TensorProto_DataType& dtype,
                     const std::vector<int64_t>& shape,
                     const std::vector<T>& value);
};

template <typename T>
std::string OnnxHelper::Constant(const std::vector<int64_t>& shape,
                                 ONNX_NAMESPACE::TensorProto_DataType dtype,
                                 std::vector<T>& value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  auto name = MapperHelper::Get()->GenName("const");
  node->add_output(name);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(name);

  int numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
    numel *= shape[i];
  }
  Assert(numel == value.size(),
         "numel and val number is not equal in Constant "
         "function.");
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data;
    data.reserve(numel);
    for (auto& i : value) {
      data.push_back(static_cast<float>(i));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data;
    data.reserve(numel);
    for (auto& i : value) {
      data.push_back(static_cast<double>(i));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data;
    data.reserve(numel);
    for (auto& i : value) {
      data.push_back(static_cast<int64_t>(i));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::BOOL) {
    bool* data = new bool[numel];
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<bool>(value[i]);
    }
    tensor->set_raw_data(std::string((const char*)(data), numel));
    delete[] data;
  } else {
    Assert(false,
           "Only support data type of BOOL/FLOAT/DOUBLE/INT64 in Constant "
           "function.");
  }
  nodes.push_back(node);
  return node->output(0);
}

template <typename T>
std::string OnnxHelper::Constant(const std::string& output,
                                 ONNX_NAMESPACE::TensorProto_DataType dtype,
                                 const std::vector<T>& value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  node->add_output(output);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(output);

  int numel = value.size();
  tensor->add_dims(numel);
  tensor->set_data_type(dtype);
  if (value.size() == 0) {
    nodes.push_back(node);
    return output;
  }
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data;
    for (auto& item : value) {
      data.push_back(static_cast<float>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data;
    for (auto& item : value) {
      data.push_back(static_cast<double>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data;
    for (auto& item : value) {
      data.push_back(static_cast<int64_t>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
    std::vector<int32_t> data;
    for (auto& item : value) {
      data.push_back(static_cast<int32_t>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::BOOL) {
    bool* data = new bool[numel];
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<bool>(value[i]);
    }
    tensor->set_raw_data(std::string((const char*)(data), numel));
    delete[] data;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT8) {
    std::vector<int8_t> data;
    data.reserve(numel);
    for (auto& i : value) {
      data.push_back(static_cast<int8_t>(i));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel));
  } else {
    Assert(false,
           "Only support data type of BOOL/FLOAT/DOUBLE/INT32/INT64/INT8 in "
           "Constant "
           "function.");
  }
  nodes.push_back(node);
  return output;
}

template <typename T>
std::string OnnxHelper::Constant(ONNX_NAMESPACE::TensorProto_DataType dtype,
                                 const std::vector<T>& value) {
  auto output = MapperHelper::Get()->GenName("helper.constant");
  return Constant(output, dtype, value);
}

template <typename T>
std::string OnnxHelper::Constant(const std::string& output,
                                 const std::vector<int64_t>& shape,
                                 ONNX_NAMESPACE::TensorProto_DataType dtype,
                                 T value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  node->add_output(output);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(output);

  int numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
    numel *= shape[i];
  }
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data(numel, static_cast<float>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data(numel, static_cast<double>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data(numel, static_cast<int64_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
    std::vector<int32_t> data(numel, static_cast<int32_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT8) {
    std::vector<int8_t> data(numel, static_cast<int8_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::BOOL) {
    bool* data = new bool[numel];
    for (size_t i = 0; i < numel; ++i) {
      data[i] = static_cast<bool>(value);
    }
    tensor->set_raw_data(std::string((const char*)(data), numel));
    delete[] data;
  } else {
    Assert(
        false,
        "Only support data type of BOOL/FLOAT/DOUBLE/INT32/INT64 in Constant "
        "function.");
  }
  nodes.push_back(node);
  return output;
}

template <typename T>
std::string OnnxHelper::Constant(const std::vector<int64_t>& shape,
                                 ONNX_NAMESPACE::TensorProto_DataType dtype,
                                 T value) {
  auto output = MapperHelper::Get()->GenName("helper.constant");
  return Constant(output, shape, dtype, value);
}

template <typename T>
std::string OnnxHelper::ConstOfShape(const std::string& input,
                                     ONNX_NAMESPACE::TensorProto_DataType dtype,
                                     T value) {
  auto output = MapperHelper::Get()->GenName("helper.constofshape");
  return ConstOfShape(input, output, dtype, value);
}

template <typename T>
std::string OnnxHelper::ConstOfShape(const std::string& input,
                                     const std::string& output,
                                     ONNX_NAMESPACE::TensorProto_DataType dtype,
                                     T value) {
  auto node = MakeNode("ConstantOfShape", {input}, {output});
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name("tensor_value");
  std::vector<int64_t> shape = {1};
  int numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
    numel *= shape[i];
  }
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data(numel, static_cast<float>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data(numel, static_cast<double>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data(numel, static_cast<int64_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
    std::vector<int32_t> data(numel, static_cast<int32_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else {
    Assert(false,
           "Only support data type of FLOAT/DOUBLE/INT64/INT32 in ConstOfShape "
           "function.");
  }
  return output;
}

template <typename T>
std::string OnnxHelper::Assign(
    const std::string& output,
    const ONNX_NAMESPACE::TensorProto_DataType& dtype,
    const std::vector<int64_t>& shape, const std::vector<T>& value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  node->add_output(output);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(output);

  int numel = std::accumulate(std::begin(shape), std::end(shape), 1,
                              std::multiplies<int64_t>());
  Assert(numel == value.size(),
         "Numel of value not satisfy the input shape while creating contant "
         "tensor.");
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
  }
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data;
    for (auto& item : value) {
      data.push_back(static_cast<float>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data;
    for (auto& item : value) {
      data.push_back(static_cast<double>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data;
    for (auto& item : value) {
      data.push_back(static_cast<int64_t>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
    std::vector<int32_t> data;
    for (auto& item : value) {
      data.push_back(static_cast<int32_t>(item));
    }
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else {
    Assert(false,
           "Only support data type of FLOAT/DOUBLE/INT32/INT64 in Constant "
           "function.");
  }
  nodes.push_back(node);
  return output;
}

template <typename T>
std::string OnnxHelper::Assign(
    const ONNX_NAMESPACE::TensorProto_DataType& dtype,
    const std::vector<int64_t>& shape, const std::vector<T>& value) {
  auto output = MapperHelper::Get()->GenName("helper.constant");
  return Assign(output, dtype, shape, value);
}

template <typename T>
bool OnnxHelper::TryGetTensorValue(const std::string& name,
                                   std::vector<T>* value) {
  for (auto iter = nodes.begin(); iter != nodes.end(); iter++) {
    auto node = *iter;
    if (node->op_type() != "Constant") {
      continue;
    }
    if (node->output(0) == name) {
      for (auto i = 0; i < node->attribute_size(); i++) {
        auto attr = node->attribute(i);
        if (attr.name() == "value") {
          auto tensor = attr.mutable_t();
          auto dtype = tensor->data_type();
          std::vector<int64_t> shape;
          for (int64_t i = 0; i < tensor->dims_size(); i++) {
            shape.push_back(tensor->dims(i));
          }
          int64_t nums = 1;
          for (auto& i : shape) nums *= i;
          value->resize(nums);
          if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
            std::vector<int64_t> val(nums, 0);
            memcpy(val.data(), tensor->raw_data().data(),
                   nums * sizeof(int64_t));
            value->assign(val.begin(), val.end());
            return true;
          } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
            std::vector<int32_t> val(nums, 0);
            memcpy(val.data(), tensor->raw_data().data(),
                   nums * sizeof(int32_t));
            value->assign(val.begin(), val.end());
            return true;
          } else if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
            std::vector<float> val(nums, 0);
            memcpy(val.data(), tensor->raw_data().data(), nums * sizeof(float));
            value->assign(val.begin(), val.end());
            return true;
          } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
            std::vector<double> val(nums, 0);
            memcpy(val.data(), tensor->raw_data().data(),
                   nums * sizeof(double));
            value->assign(val.begin(), val.end());
            return true;
          } else {
            P2OLogger() << "[WARNING] OnnxHelper function TryGetTensorValue "
                           "only support get int64_t/int32_t/float/double "
                           "value from Constant now."
                        << std::endl;
            return false;
          }
        }
      }
    }
  }
  return false;
}

}  // namespace paddle2onnx
