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

#include "paddle2onnx/mapper/nn/pool3d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(pool3d, Pool3dMapper)
REGISTER_MAPPER(max_pool3d_with_index, Pool3dMapper)

bool Pool3dMapper::IsSameSpan(const int64_t& in_size, const int64_t& out_size) {
  std::vector<int64_t> spans;
  spans.reserve(out_size);
  for (auto i = 0; i < out_size; ++i) {
    int64_t start = std::floor(i * (in_size / out_size));
    int64_t end = std::ceil((i + 1) * (in_size / out_size));
    spans.push_back(end - start);
  }
  std::sort(spans.begin(), spans.end());
  return spans[0] == spans[spans.size() - 1];
}

void Pool3dMapper::AdaptivePool(const std::vector<TensorInfo>& input_info,
                                const std::vector<TensorInfo>& output_info) {
  int64_t input_d = input_info[0].shape[2];
  int64_t input_h = input_info[0].shape[3];
  int64_t input_w = input_info[0].shape[4];
  int64_t output_d = output_info[0].shape[2];
  int64_t output_h = output_info[0].shape[3];
  int64_t output_w = output_info[0].shape[4];
  int64_t stride_d = std::floor(input_d / output_d);
  int64_t stride_h = std::floor(input_h / output_h);
  int64_t stride_w = std::floor(input_w / output_w);
  int64_t kernel_d = input_d - (output_d - 1) * stride_d;
  int64_t kernel_h = input_h - (output_h - 1) * stride_h;
  int64_t kernel_w = input_w - (output_w - 1) * stride_w;
  std::string onnx_pool_type;
  if (OpType() == "max_pool3d_with_index") {
    onnx_pool_type = "MaxPool";
  } else {
    auto iter = op_mapper_.find(pooling_type_);
    onnx_pool_type = iter->second[0];
  }
  std::shared_ptr<ONNX_NAMESPACE::NodeProto>* node_ptr;
  auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                 P2ODataType::FP32);
  auto node = helper_->MakeNode(onnx_pool_type, {input});
  helper_->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                    output_info[0].dtype);
  std::vector<int64_t> kernel_size = {kernel_d, kernel_h, kernel_w};
  AddAttribute(node, "kernel_shape", kernel_size);
  std::vector<int64_t> strides = {stride_d, stride_h, stride_w};
  AddAttribute(node, "strides", strides);

  if (helper_->GetOpsetVersion() > 10) {
    AddAttribute(node, "ceil_mode", static_cast<int64_t>(ceil_mode_));
  }

  std::string auto_pad = "NOTSET";
  if (padding_algorithm_ == "SAME") {
    auto_pad = "SAME_UPPER";
  } else if (padding_algorithm_ == "VALID") {
    auto_pad = "VALID";
  }
  AddAttribute(node, "auto_pad", auto_pad);
  if (pooling_type_ == "avg") {
    AddAttribute(node, "count_include_pad", static_cast<int64_t>(exclusive_));
  }
}

void Pool3dMapper::NoAdaptivePool(const std::vector<TensorInfo>& input_info,
                                  const std::vector<TensorInfo>& output_info) {
  std::vector<int64_t> input_shape = input_info[0].shape;
  if (pads_.size() == 3) {
    pads_.push_back(pads_[0]);
    pads_.push_back(pads_[1]);
    pads_.push_back(pads_[2]);
  } else if (pads_.size() == 6) {
    std::vector<int64_t> index = {0, 2, 4, 1, 3, 5};
    std::vector<int64_t> copy = pads_;
    for (auto i = 0; i < index.size(); ++i) {
      pads_[i] = copy[index[i]];
    }
  }
  if (input_shape[2] > 0 && input_shape[2] + pads_[0] < k_size_[0]) {
    k_size_[0] = input_shape[2] + pads_[0];
  }
  if (input_shape[3] > 0 && input_shape[3] + pads_[1] < k_size_[1]) {
    k_size_[1] = input_shape[3] + pads_[1];
  }
  if (input_shape[4] > 0 && input_shape[4] + pads_[2] < k_size_[2]) {
    k_size_[2] = input_shape[4] + pads_[2];
  }

  int64_t max_ksize = *std::max_element(std::begin(k_size_), std::end(k_size_));
  int64_t max_pads = *std::max_element(std::begin(pads_), std::end(pads_));
  auto input_x = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
  if (max_ksize <= max_pads) {
    std::vector<int64_t> onnx_paddings = {0, 0, pads_[0], pads_[1], pads_[2],
                                          0, 0, pads_[3], pads_[4], pads_[5]};
    std::vector<std::string> inputs_names = {input_x};
    if (helper_->GetOpsetVersion() >= 11) {
      std::string paddings_node =
          helper_->Constant(GetOnnxDtype(P2ODataType::INT64), onnx_paddings);
      inputs_names.push_back(paddings_node);
      std::vector<float> val = {0.0};
      std::string val_node =
          helper_->Constant(GetOnnxDtype(P2ODataType::FP32), val);
      inputs_names.push_back(val_node);
    }
    auto node = helper_->MakeNode("Pad", inputs_names);
    std::string mode = "constant";
    AddAttribute(node, "mode", mode);
    if (helper_->GetOpsetVersion() < 11) {
      AddAttribute(node, "pads", onnx_paddings);
      float val = 0.0;
      AddAttribute(node, "value", val);
    }
    input_x = node->output(0);
    pads_.clear();
    pads_.resize(6, 0);
  }
  std::string onnx_pool_type;
  if (OpType() == "max_pool3d_with_index") {
    onnx_pool_type = "MaxPool";
  } else {
    auto iter = op_mapper_.find(pooling_type_);
    onnx_pool_type = iter->second[0];
  }
  auto node = helper_->MakeNode(onnx_pool_type, {input_x});
  helper_->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                    output_info[0].dtype);

  AddAttribute(node, "kernel_shape", k_size_);
  AddAttribute(node, "strides", strides_);
  std::string auto_pad = "NOTSET";
  if (padding_algorithm_ == "SAME") {
    auto_pad = "SAME_UPPER";
    AddAttribute(node, "auto_pad", auto_pad);
  } else if (padding_algorithm_ == "VALID") {
    auto_pad = "VALID";
    AddAttribute(node, "auto_pad", auto_pad);
  } else {
    AddAttribute(node, "pads", pads_);
  }
  if (OpType() != "max_pool3d_with_index" && helper_->GetOpsetVersion() >= 10) {
    AddAttribute(node, "ceil_mode", static_cast<int64_t>(ceil_mode_));
  }
  if (pooling_type_ == "avg") {
    AddAttribute(node, "count_include_pad", static_cast<int64_t>(exclusive_));
  }
}

int32_t Pool3dMapper::GetMinOpset(bool verbose) {
  // NHWC is not supported
  if (data_format_ == "NDHWC") {
    Error() << "NDHWC format is not supported." << std::endl;
    return -1;
  }
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  if (global_pooling_ ||
      (k_size_[0] == 1 && k_size_[1] == 1 && k_size_[2] == 1)) {
    if (ceil_mode_) {
      Logger(verbose, 10) << "While ceil_model is True, " << RequireOpset(10)
                          << std::endl;
      return 10;
    }
    return 7;
  }

  if (adaptive_) {
    for (auto one_input : input_info) {
      for (auto i = 2; i < one_input.shape.size(); ++i) {
        if (one_input.shape[i] == -1) {
          Error() << "Adaptive only support static input shape." << std::endl;
          return -1;
        }
      }
    }
    int64_t input_d = input_info[0].shape[2];
    int64_t input_h = input_info[0].shape[3];
    int64_t input_w = input_info[0].shape[4];
    int64_t output_d = output_info[0].shape[2];
    int64_t output_h = output_info[0].shape[3];
    int64_t output_w = output_info[0].shape[4];
    if (!IsSameSpan(input_h, output_h) || !IsSameSpan(input_w, output_w) ||
        !IsSameSpan(input_d, output_d)) {
      Error() << "Cannot convert adaptive pool with input_size: " << input_d
              << " " << input_h << " " << input_w
              << " output_size: " << output_d << " " << output_h << " "
              << output_w << std::endl;
      return -1;
    }
  }
  if (OpType() == "max_pool3d_with_index") {
    return 9;
  }
  auto iter = op_mapper_.find(pooling_type_);
  if (op_mapper_.end() == iter) {
    Error() << "Cannot find " << pooling_type_ << " in pool op_mapper."
            << std::endl;
    return -1;
  }

  if (ceil_mode_) {
    Logger(verbose, 10) << "While ceil_model is True, " << RequireOpset(10)
                        << std::endl;
    return 10;
  }
  return 7;
}

void Pool3dMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  bool is_1x1_kernel = true;
  for (auto i : k_size_) {
    if (i != 1) {
      is_1x1_kernel = false;
    }
  }

  if (global_pooling_ || (adaptive_ && is_1x1_kernel)) {
    std::string onnx_pool_type;
    if (OpType() == "max_pool3d_with_index") {
      onnx_pool_type = "GlobalMaxPool";
    } else {
      auto iter = op_mapper_.find(pooling_type_);
      onnx_pool_type = iter->second[1];
    }
    auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
    auto output = helper_->MakeNode(onnx_pool_type, {input})->output(0);
    helper_->AutoCast(output, output_info[0].name, P2ODataType::FP32,
                      output_info[0].dtype);
  } else if (adaptive_) {
    AdaptivePool(input_info, output_info);
  } else {
    NoAdaptivePool(input_info, output_info);
  }
}

}  // namespace paddle2onnx
