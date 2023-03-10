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

class ElementwiseMapper : public Mapper {
 public:
  ElementwiseMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("axis", &axis_);

    op_mapper_["elementwise_add"] = "Add";
    op_mapper_["elementwise_sub"] = "Sub";
    op_mapper_["elementwise_div"] = "Div";
    op_mapper_["elementwise_mul"] = "Mul";
    op_mapper_["elementwise_min"] = "Min";
    op_mapper_["elementwise_max"] = "Max";
    op_mapper_["elementwise_pow"] = "Pow";
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();

 private:
  std::map<std::string, std::string> op_mapper_;
  int64_t axis_;
};

class ElementWiseModMapper : public Mapper {
 public:
  ElementWiseModMapper(const PaddleParser& p, OnnxHelper* helper,
                       int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) {
    Logger(verbose, 10) << RequireOpset(10) << std::endl;
    return 10;
  }

  void Opset10();
};

class ElementWiseFloordivMapper : public Mapper {
 public:
  ElementWiseFloordivMapper(const PaddleParser& p, OnnxHelper* helper,
                            int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("axis", &axis_);
  }

  void Opset7();

 private:
  int64_t axis_;
};

}  // namespace paddle2onnx
