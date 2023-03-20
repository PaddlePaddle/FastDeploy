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

#include "paddle2onnx/mapper/quantize/dequantize_linear.h"

namespace paddle2onnx {
REGISTER_MAPPER(dequantize_linear, DequantizeLinearMapper)

int32_t DequantizeLinearMapper::GetMinOpset(bool verbose) {
  if (!IsConstantInput("Scale")) {
    Error() << "Input `Scale` requires to be a constant tensor." << std::endl;
    return -1;
  }
  std::vector<float> scales;
  if (!TryGetInputValue("Scale", &scales)) {
    Error() << "Failed to read tensor value of `Scale`." << std::endl;
    return -1;
  }
  if (bit_length_ != 8) {
    Error() << "Only support bit_length = 8." << std::endl;
    return -1;
  }
  if (scales.size() > 1) {
    auto x_info = GetInput("X");
    if (x_info[0].shape[quant_axis_] != scales.size()) {
      Error() << "Scale size must equal to the size of input quantize axis."
              << std::endl;
      return -1;
    }
    Logger(verbose, 13) << "While size of scales greater than 1, "
                        << RequireOpset(13) << std::endl;
    return 13;
  }
  auto x_info = GetInput("X");
  auto x_shape = x_info[0].shape;
  if (x_shape.size() == 2) {
    if (quant_axis_ != 1) {
      Error() << "When the rank of input is 2, the attribute quant_axis "
                 "requires to be 1."
              << std::endl;
      return -1;
    }
  } else if (x_shape.size() == 4) {
    if (!(quant_axis_ == 1 || quant_axis_ == 0)) {
      Error() << "When the rank of input is 4, the attribute quant_axis "
                 "requires to be 0 or 1."
              << std::endl;
      return -1;
    }
  }

  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void DequantizeLinearMapper::ConvertInt8ToFp32(
    const std::vector<float> &onnx_scales, std::vector<float> *weight) {
  auto x_info = GetInput("X");
  auto x_shape = x_info[0].shape;
  if (x_shape.size() == 2) {
    for (auto j = 0; j < x_shape[1]; ++j) {
      float scale_value = 0;
      if (onnx_scales.size() == 1) {
        scale_value = onnx_scales[0];
      } else {
        scale_value = onnx_scales[j];
      }
      for (auto i = 0; i < x_shape[0]; ++i) {
        auto offset = i * x_shape[1] + j;
        (*weight)[offset] *= scale_value;
      }
    }
  } else if (x_shape.size() == 4) {
    if (quant_axis_ == 0) {
      auto inner_offset = 1;
      for (auto i : x_shape) {
        inner_offset *= i;
      }
      inner_offset /= x_shape[0];
      for (int i = 0; i < x_shape[0]; ++i) {
        float scale_value = 0;
        if (onnx_scales.size() == 1) {
          scale_value = onnx_scales[0];
        } else {
          scale_value = onnx_scales[i];
        }
        for (auto j = 0; j < inner_offset; ++j) {
          auto offset = i * inner_offset + j;
          (*weight)[offset] *= scale_value;
        }
      }
    } else {
      auto inner_offset = x_shape[2] * x_shape[3];
      auto outter_offset = x_shape[1] * inner_offset;
      for (auto i = 0; i < x_shape[0]; ++i) {
        for (auto j = 0; j < x_shape[1]; ++j) {
          float scale_value = 0;
          if (onnx_scales.size() == 1) {
            scale_value = onnx_scales[0];
          } else {
            scale_value = onnx_scales[j];
          }
          for (auto k = 0; k < inner_offset; k++) {
            auto offset = i * outter_offset + j * inner_offset + k;
            (*weight)[offset] *= scale_value;
          }
        }
      }
    }
  }
}

void DequantizeLinearMapper::Opset10() {
  auto x_info = GetInput("X");
  auto x_shape = x_info[0].shape;
  std::vector<float> scales;
  Assert(TryGetInputValue("Scale", &scales),
         "Failed to read tensor value of `Scale`.");
  std::vector<float> onnx_scales;
  onnx_scales.reserve(scales.size());
  for (auto &i : scales) {
    onnx_scales.push_back(i / 127);
  }
  std::vector<int64_t> onnx_zeros(onnx_scales.size(), 0);
  std::string scale_node, zero_node;
  if (onnx_zeros.size() == 1) {
    scale_node = helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                   onnx_scales[0]);
    zero_node =
        helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8, onnx_zeros[0]);
  } else {
    scale_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, onnx_scales);
    zero_node =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, onnx_zeros);
  }

  std::vector<float> weight;
  TryGetInputValue("X", &weight);
  if (weight.empty()) {
    auto node = helper_->MakeNode("DequantizeLinear",
                                  {x_info[0].name, scale_node, zero_node},
                                  {GetOutput("Y")[0].name});
    if (helper_->GetOpsetVersion() >= 13) {
      AddAttribute(node, "axis", quant_axis_);
    }
    QuantizeInfo quantize_info(onnx_scales, onnx_zeros, scale_node, zero_node,
                               quant_axis_);
    helper_->quantize_info[GetOutput("Y")[0].name] = quantize_info;
    return;
  }
  ConvertInt8ToFp32(onnx_scales, &weight);

  QuantizeInfo quantize_info(onnx_scales, onnx_zeros, scale_node, zero_node,
                             quant_axis_);
  helper_->quantize_info[x_info[0].name] = quantize_info;
  Weight fp32_weight;
  fp32_weight.set(P2ODataType::FP32, x_shape, weight);
  helper_->updated_params[x_info[0].name] = fp32_weight;
  auto node = helper_->MakeNode("QuantizeLinear",
                                {x_info[0].name, scale_node, zero_node});
  if (helper_->GetOpsetVersion() >= 13) {
    AddAttribute(node, "axis", quant_axis_);
  }
  auto dq_node = helper_->MakeNode("DequantizeLinear",
                                   {node->output(0), scale_node, zero_node},
                                   {GetOutput("Y")[0].name});
  if (helper_->GetOpsetVersion() >= 13) {
    AddAttribute(dq_node, "axis", quant_axis_);
  }
}
}  // namespace paddle2onnx