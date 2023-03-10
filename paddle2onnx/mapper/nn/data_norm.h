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

class DataNormMapper : public Mapper {
 public:
  DataNormMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("data_layout", &data_layout_);
    GetAttr("epsilon", &epsilon_);
    if (HasAttr("slot_dim")) {
      GetAttr("slot_dim", &slot_dim_);
    }
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();

 private:
  std::string data_layout_;
  float epsilon_;
  int64_t slot_dim_ = -1;
};

}  // namespace paddle2onnx
