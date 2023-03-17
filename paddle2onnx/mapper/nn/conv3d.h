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

class Conv3dMapper : public Mapper {
 public:
  Conv3dMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("groups", &groups_);
    GetAttr("dilations", &dilations_);
    GetAttr("strides", &strides_);
    GetAttr("paddings", &paddings_);
    GetAttr("padding_algorithm", &padding_algorithm_);
    GetAttr("data_format", &data_format_);
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();

 private:
  std::vector<int64_t> dilations_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> paddings_;
  std::string padding_algorithm_;
  std::string data_format_;
  int64_t groups_;
};

}  // namespace paddle2onnx
