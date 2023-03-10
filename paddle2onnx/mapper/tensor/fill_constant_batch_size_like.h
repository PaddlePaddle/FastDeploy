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
#include <map>
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class FillConstantBatchSizeLikeMapper : public Mapper {
 public:
  FillConstantBatchSizeLikeMapper(const PaddleParser& p, OnnxHelper* helper,
                                  int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("dtype", &dtype_);
    GetAttr("value", &value_);
    GetAttr("shape", &shape_);
    GetAttr("str_value", &str_value_);
    GetAttr("input_dim_idx", &input_dim_idx_);
    GetAttr("output_dim_idx", &output_dim_idx_);
  }

  int32_t GetMinOpset(bool verbose = true);
  void Opset7();

 private:
  int64_t dtype_;
  float value_;
  std::string str_value_;
  int64_t input_dim_idx_;
  int64_t output_dim_idx_;
  std::vector<int64_t> shape_;
};

}  // namespace paddle2onnx
