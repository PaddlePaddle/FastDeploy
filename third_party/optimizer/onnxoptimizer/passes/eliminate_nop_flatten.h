/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopFlatten final : public PredicateBasedPass {
  explicit EliminateNopFlatten()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_flatten";
  }

  bool patternMatchPredicate(Node *node) override {
    if (node->kind() != Symbol("Flatten")) {
      return false;
    }
    const Value *input = node->input();
    if (!input->has_sizes()) {
      return false;
    }
    const auto input_shape = input->sizes();
    const int axis = node->hasAttribute(kaxis) ? node->i(kaxis) : 1;
    if (input_shape.size() == 2) {
      if (axis == 1 || axis == -1) {
        return true;
      }
      if (input_shape[0].is_int && input_shape[0].dim == 1 && axis == 0) {
        return true;
      }
    }

    return false;
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
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
