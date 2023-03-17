#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <cstring>
#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"
#include "paddle2onnx/optimizer/paddle2onnx_optimizer.h"

namespace paddle2onnx {

int32_t GetDataTypeFromOnnx(int dtype) {
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    return 0;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    return 1;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::UINT8) {
    return 2;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT8) {
    return 3;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT32) {
    return 4;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    return 5;
  } else if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    return 6;
  }
  Assert(false, "Only support float/double/uint8/int32/int64/float16 in OnnxReader.");
  return -1;
}

OnnxReader::OnnxReader(const char* model_buffer, int buffer_size) {
  ONNX_NAMESPACE::ModelProto model;
  std::string content(model_buffer, model_buffer + buffer_size);
  model.ParseFromString(content);

  std::set<std::string> initializer_names;
  for (auto i = 0; i < model.graph().initializer_size(); ++i) {
    initializer_names.insert(model.graph().initializer(i).name());
  }

  num_outputs = model.graph().output_size();
  Assert(num_outputs <= 100,
         "The number of outputs is exceed 100, unexpected situation.");

  num_inputs = 0;
  for (int i = 0; i < model.graph().input_size(); ++i) {
    if (initializer_names.find(model.graph().input(i).name()) !=
        initializer_names.end()) {
      continue;
    }
    num_inputs += 1;
    Assert(num_inputs <= 100,
           "The number of inputs is exceed 100, unexpected situation.");

    inputs[i].dtype =
        GetDataTypeFromOnnx(model.graph().input(i).type().tensor_type().elem_type());
    std::strcpy(inputs[i].name, model.graph().input(i).name().c_str());
    auto& shape = model.graph().input(i).type().tensor_type().shape();
    int dim_size = shape.dim_size();
    inputs[i].rank = dim_size;
    inputs[i].shape = new int64_t[dim_size];
    for (int j = 0; j < dim_size; ++j) {
      inputs[i].shape[j] = static_cast<int64_t>(shape.dim(j).dim_value());
      if (inputs[i].shape[j] <= 0) {
        inputs[i].shape[j] = -1;
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    std::strcpy(outputs[i].name, model.graph().output(i).name().c_str());
    outputs[i].dtype =
        GetDataTypeFromOnnx(model.graph().output(i).type().tensor_type().elem_type());
    auto& shape = model.graph().output(i).type().tensor_type().shape();
    int dim_size = shape.dim_size();
    outputs[i].rank = dim_size;
    outputs[i].shape = new int64_t[dim_size];
    for (int j = 0; j < dim_size; ++j) {
      outputs[i].shape[j] = static_cast<int64_t>(shape.dim(j).dim_value());
      if (outputs[i].shape[j] <= 0) {
        outputs[i].shape[j] = -1;
      }
    }
  }
}

bool RemoveMultiClassNMS(const char* model_buffer, int buffer_size,
                         char** out_model, int* out_model_size) {
  ONNX_NAMESPACE::ModelProto model;
  std::string content(model_buffer, model_buffer + buffer_size);
  model.ParseFromString(content);
  auto* graph = model.mutable_graph();
  int nms_index = -1;
  std::vector<std::string> inputs;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).op_type() == "MultiClassNMS") {
      nms_index = -1;
      for (int j = 0; j < graph->node(i).input_size(); ++j) {
        inputs.push_back(graph->node(i).input(j));
      }
      break;
    }
  }
  graph->clear_output();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto output = graph->add_output();
    output->set_name(inputs[i]);
    auto type_proto = output->mutable_type();
    auto tensor_type_proto = type_proto->mutable_tensor_type();
    tensor_type_proto->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    auto shape = tensor_type_proto->mutable_shape();
    shape->add_dim()->set_dim_value(-1);
    shape->add_dim()->set_dim_value(-1);
    shape->add_dim()->set_dim_value(-1);
  }
  auto optimized_model = ONNX_NAMESPACE::optimization::OptimizeOnnxModel(model);
  *out_model_size = optimized_model.ByteSizeLong();
  *out_model = new char[*out_model_size];
  optimized_model.SerializeToArray(*out_model, *out_model_size);
  return true;
}

}  // namespace paddle2onnx
