//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Before:
//   X = Constant() all elements equal to 1, shape is () or (1,)
//   Y = Tensor()
//   C = X * Y
// After:
//   C = Identity(Y)

#include <numeric>
#include <cmath>
#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ReplaceMulToIdentity final : public PredicateBasedPass {
  explicit ReplaceMulToIdentity()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "replace_mul_to_identity";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kMul &&
           (node->inputs()[0]->node()->kind() == kConstant || node->inputs()[1]->node()->kind() == kConstant); 
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* mul_node = n;
    Node* mul_ipt_0 = n->inputs()[0]->node();
    Node* mul_ipt_1 = n->inputs()[1]->node();

    if (mul_ipt_0->kind() == kConstant) {
      auto scale = mul_ipt_0->t(kvalue);
      if (scale.sizes().size() == 1 && scale.sizes()[0] != 1) {
        return false;
      }
      if (scale.sizes().size() > 1) {
        return false;
      }
      const auto& float_data = scale.floats();
      if (float_data.size() > 0 && fabs(float_data[0] - 1.0) > 1e-05) {
        return false;
      }
      const auto& double_data = scale.doubles();
      if (double_data.size() > 0 && fabs(double_data[0] - 1.0) > 1e-05) {
        return false;
      }
      const auto& int32_data = scale.int32s();
      if (int32_data.size() > 0 && int32_data[0] != 1) {
        return false;
      }
      const auto& int64_data = scale.int64s();
      if (int64_data.size() > 0 && int64_data[0] != 1) {
        return false;
      }
      if (float_data.size() == 0 && double_data.size() == 0 && int32_data.size() == 0 && int64_data.size() == 0) {
        return false;
      }
      if (!tryReplacingAllUsesWith(mul_node->output(), mul_node->inputs()[1])) {
        return false;
      }
    } else {
      auto scale = mul_ipt_1->t(kvalue);
      if (scale.sizes().size() == 1 && scale.sizes()[0] != 1) {
        return false;
      }
      if (scale.sizes().size() > 1) {
        return false;
      }
      const auto& float_data = scale.floats();
      if (float_data.size() > 0 && fabs(float_data[0] - 1.0) > 1e-05) {
        return false;
      }
      const auto& double_data = scale.doubles();
      if (double_data.size() > 0 && fabs(double_data[0] - 1.0) > 1e-05) {
        return false;
      }
      const auto& int32_data = scale.int32s();
      if (int32_data.size() > 0 && int32_data[0] != 1) {
        return false;
      }
      const auto& int64_data = scale.int64s();
      if (int64_data.size() > 0 && int64_data[0] != 1) {
        return false;
      }
      if (float_data.size() == 0 && double_data.size() == 0 && int32_data.size() == 0 && int64_data.size() == 0) {
        return false;
      }
      if (!tryReplacingAllUsesWith(mul_node->output(), mul_node->inputs()[0])) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
