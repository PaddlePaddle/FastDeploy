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
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class DequantizeLinearMapper : public Mapper {
 public:
  DequantizeLinearMapper(const PaddleParser& p, OnnxHelper* helper,
                         int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("quant_axis", &quant_axis_);
    GetAttr("bit_length", &bit_length_);
    if (quant_axis_ == -1) {
      quant_axis_ = 1;
    }
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset10();

 private:
  void ConvertInt8ToFp32(const std::vector<float>& onnx_scales,
                         std::vector<float>* weight);
  int64_t quant_axis_ = 1;
  int64_t bit_length_ = 8;
};

}  // namespace paddle2onnx
