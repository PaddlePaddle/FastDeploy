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
//   B = Unsqueeze(Constant, axes)
// After:
//   B = Constant (Constant with new shape)

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConstantUnsqueeze final : public PredicateBasedPass {
  explicit FuseConstantUnsqueeze()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_constant_unsqueeze"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kUnsqueeze &&
           node->inputs()[0]->node()->kind() == kConstant;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Constant is only used by Reshape
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* unsqueeze = n;
    Node* constant = n->inputs()[0]->node();

    // Process 'axes' data
    std::vector<int64_t> axes;
    if (unsqueeze->hasAttribute(kaxes)) {
      // opset 13 below
      axes = unsqueeze->is(kaxes);
    } else {
      // opset 13 and above - first check if 'unsqueeze' has 'axes' input
      // constant
      if (unsqueeze->inputs()[1]->node()->kind() != kConstant) {
        return false;
      }
      if (unsqueeze->inputs()[1]->uses().size() > 1) {
        return false;
      }
      Node* axes_const = unsqueeze->inputs()[1]->node();
      Tensor t = axes_const->t(kvalue);
      axes = ParseData<int64_t>(&t);
    }

    Tensor t = constant->t(kvalue);
    const auto& ori_size = t.sizes();
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i] < 0) {
        axes[i] = axes[i] + ori_size.size() + i + 1;
      }
    }

    std::vector<int64_t> new_size(ori_size.begin(), ori_size.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      new_size.insert(new_size.begin() + axes[i], 1);
    }

    t.sizes().clear();
    t.sizes().insert(t.sizes().begin(), new_size.begin(),
                     new_size.begin() + new_size.size());
    constant->t_(kvalue, std::move(t));

    // update constant node
    constant->output()->setSizes(unsqueeze->output()->sizes());
    constant->output()->setElemType(unsqueeze->output()->elemType());
    const bool replacing_success =
        tryReplacingAllUsesWith(unsqueeze->output(), unsqueeze->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
