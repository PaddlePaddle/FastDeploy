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

class OneHotV2Mapper : public Mapper {
 public:
  OneHotV2Mapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                 int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("allow_out_of_range", &allow_out_of_range_);
    GetAttr("depth", &depth_);
    GetAttr("dtype", &dtype_);
  }
  int32_t GetMinOpset(bool verbose);
  void Opset9();

 private:
  bool allow_out_of_range_;
  int64_t depth_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
