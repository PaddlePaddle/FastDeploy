/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 A = Constant()
// After:
//	 A is in the initializer list
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: A is the output of a Constant node
#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ExtractConstantToInitializer final : public PredicateBasedPass {
  explicit ExtractConstantToInitializer()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "extract_constant_to_initializer";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConstant;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Tensor t = node->t(kvalue);
    Value* new_init;
    if (node->output()->has_unique_name() &&
        std::find(graph.outputs().rbegin(), graph.outputs().rend(),
                  node->output()) == graph.outputs().rend()) {
      new_init = graph.addInitializerAndInput(t, node->output()->uniqueName());
      node->output()->setUniqueName(
          ONNX_NAMESPACE::to_string(graph.getNextUnique()), false);
    } else {
      // the unique_name will be set in `replaceAllUsesWith` if
      // node->output() is in graph output
      new_init = graph.addInitializerAndInput(t);
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), new_init);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
