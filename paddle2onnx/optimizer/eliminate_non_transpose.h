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

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNonTranspose final : public PredicateBasedPass {
  explicit EliminateNonTranspose()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_non_transpose"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kTranspose;
  }
  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (node->hasAttribute(kperm)) {
      auto perm = node->is(kperm);
      for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != i) {
          return false;
        }
      }
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->input());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
