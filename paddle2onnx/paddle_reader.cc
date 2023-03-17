#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <cstring>
#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"
#include "paddle2onnx/parser/parser.h"

namespace paddle2onnx {

int32_t GetDataTypeFromPaddle(int dtype) {
  if (dtype == P2ODataType::FP32) {
    return 0;
  } else if (dtype == P2ODataType::FP64) {
    return 1;
  } else if (dtype == P2ODataType::UINT8) {
    return 2;
  } else if (dtype == P2ODataType::INT8) {
    return 3;
  } else if (dtype == P2ODataType::INT32) {
    return 4;
  } else if (dtype == P2ODataType::INT64) {
    return 5;
  }
  Assert(false, "Only support float/double/uint8/int32/int64 in PaddleReader.");
  return -1;
}

PaddleReader::PaddleReader(const char* model_buffer, int buffer_size) {
  PaddleParser parser;
  Assert(parser.Init(model_buffer, buffer_size),
         "Failed to parse PaddlePaddle model.");

  num_inputs = parser.inputs.size();
  num_outputs = parser.outputs.size();
  for (int i = 0; i < num_inputs; ++i) {
    std::strcpy(inputs[i].name, parser.inputs[i].name.c_str());
    inputs[i].rank = parser.inputs[i].Rank();
    inputs[i].shape = new int64_t[inputs[i].rank];
    for (int j = 0; j < inputs[i].rank; ++j) {
      inputs[i].shape[j] = parser.inputs[i].shape[j];
    }
    inputs[i].dtype = GetDataTypeFromPaddle(parser.inputs[i].dtype);
  }

  for (int i = 0; i < num_outputs; ++i) {
    std::strcpy(outputs[i].name, parser.outputs[i].name.c_str());
    outputs[i].rank = parser.outputs[i].Rank();
    outputs[i].shape = new int64_t[outputs[i].rank];
    for (int j = 0; j < outputs[i].rank; ++j) {
      outputs[i].shape[j] = parser.outputs[i].shape[j];
    }
    outputs[i].dtype = GetDataTypeFromPaddle(parser.outputs[i].dtype);
  }
  for (size_t i = 0; i < parser.NumOfOps(0); ++i) {
    if (parser.GetOpDesc(0, i).type().find("quantize") != std::string::npos) {
      is_quantize_model = true;
      break;
    }
  }
  for (size_t i = 0; i < parser.NumOfOps(0); ++i) {
    if (parser.GetOpDesc(0, i).type().find("multiclass_nms3") != std::string::npos) {
      has_nms = true;
      auto& op = parser.GetOpDesc(0, i);
      parser.GetOpAttr(op, "background_label", &nms_params.background_label);
      parser.GetOpAttr(op, "keep_top_k", &nms_params.keep_top_k);
      parser.GetOpAttr(op, "nms_eta", &nms_params.nms_eta);
      parser.GetOpAttr(op, "nms_threshold", &nms_params.nms_threshold);
      parser.GetOpAttr(op, "score_threshold", &nms_params.score_threshold);
      parser.GetOpAttr(op, "nms_top_k", &nms_params.nms_top_k);
      parser.GetOpAttr(op, "normalized", &nms_params.normalized);
      break;
    }
  }
}

}  // namespace paddle2onnx
