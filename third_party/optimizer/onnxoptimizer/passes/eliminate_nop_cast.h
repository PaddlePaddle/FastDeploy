/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopCast final : public PredicateBasedPass {
  explicit EliminateNopCast()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_cast";
  }

  bool patternMatchPredicate(Node* node) override {
    return (node->kind() == kCast && node->hasAttribute(kto) &&
            node->input()->elemType() == node->i(kto));
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
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
