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
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class Pool2dMapper : public Mapper {
 public:
  Pool2dMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    op_mapper_["max"] = {"MaxPool", "GlobalMaxPool"};
    op_mapper_["avg"] = {"AveragePool", "GlobalAveragePool"};
    GetAttr("global_pooling", &global_pooling_);
    GetAttr("adaptive", &adaptive_);
    GetAttr("strides", &strides_);
    GetAttr("paddings", &pads_);
    if (OpType() != "max_pool2d_with_index") {
      GetAttr("pooling_type", &pooling_type_);
      GetAttr("data_format", &data_format_);
      GetAttr("ceil_mode", &ceil_mode_);
      GetAttr("padding_algorithm", &padding_algorithm_);
      GetAttr("exclusive", &exclusive_);
      exclusive_ = !exclusive_;
    }
  }
  int32_t GetMinOpset(bool verbose = false);
  void Opset7();
  void ExportAsCustomOp();
  bool IsExportAsCustomOp();

 private:
  bool IsSameSpan(const int64_t& in_size, const int64_t& out_size);
  void AdaptivePool(const std::vector<TensorInfo>& input_info,
                    const std::vector<TensorInfo>& output_info);
  void NoAdaptivePool(const std::vector<TensorInfo>& input_info,
                      const std::vector<TensorInfo>& output_info);
  bool ceil_mode_;
  bool global_pooling_;
  bool adaptive_;
  bool exclusive_;
  std::string data_format_;
  std::string pooling_type_;
  std::string padding_algorithm_;
  std::vector<int64_t> k_size_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::map<std::string, std::vector<std::string>> op_mapper_;
};

}  // namespace paddle2onnx
