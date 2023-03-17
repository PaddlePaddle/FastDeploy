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

#include "paddle2onnx/mapper/quantize/quantize_linear.h"

namespace paddle2onnx {
REGISTER_MAPPER(quantize_linear, QuantizeLinearMapper)

int32_t QuantizeLinearMapper::GetMinOpset(bool verbose) {
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
  if (round_type_ != 0) {
    Error() << "The round_type attr of quantize_linear must be 0." << std::endl;
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
  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void QuantizeLinearMapper::Opset10() {
  auto x_info = GetInput("X");
  std::vector<float> scales;
  Assert(TryGetInputValue("Scale", &scales),
         "Failed to read tensor value of `Scale`.");
  std::vector<float> onnx_scales;
  onnx_scales.reserve(scales.size());
  for (auto i : scales) {
    onnx_scales.push_back(i / 127);
  }
  std::vector<int64_t> onnx_zeros(onnx_scales.size(), 0);

  std::string scale_node, zero_node;
  if (onnx_scales.size() == 1) {
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

  auto node = helper_->MakeNode("QuantizeLinear",
                                {x_info[0].name, scale_node, zero_node},
                                {GetOutput("Y")[0].name});
  if (helper_->GetOpsetVersion() >= 13) {
    AddAttribute(node, "axis", quant_axis_);
  }
  QuantizeInfo quantize_info(onnx_scales, onnx_zeros, scale_node, zero_node,
                             quant_axis_);
  helper_->quantize_info[x_info[0].name] = quantize_info;
}
}  // namespace paddle2onnx