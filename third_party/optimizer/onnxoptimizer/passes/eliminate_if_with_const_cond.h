/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// This optimization works well especially when used together with
// constant folding (onnx-simplifier), for example, the if node
// introduced by PyTorch squeeze op will be eliminated when the input
// shape is known.
// Ideally eliminate_if_with_const_cond + eliminate_deadend + constant
// folding can be replaced by the more powerful sparse conditional
// constant propagation, which obviously cannot be implemented in
// the current optimizer framework.

struct EliminateIfWithConstCond final : public PredicateBasedPass {
  explicit EliminateIfWithConstCond()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_if_with_const_cond";
  }

  // step 1: find "if" node with constant cond (i.e. const true or false)
  bool patternMatchPredicate(Node *node) override {
    if (node->kind() == kIf) {
      const auto cond_value = node->input();
      if ((cond_value->node()->kind() == kConstant ||
           cond_value->node()->kind() == kParam)) {
        return true;
      }
    }
    return false;
  }

  // step 2: inline the subgraph (for example, inline then_branch when cond ===
  // true)
  //         by re-creating all subgraph nodes in parent graph
  //         note: handle captured value
  // step 3: Delete "if" node itself
  bool runTransform(Node *if_node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const auto cond_value = if_node->input();
    Tensor cond_tensor;
    if (cond_value->node()->kind() == kConstant) {
      cond_tensor = cond_value->node()->t(kvalue);
    } else {
      cond_tensor = *graph.getInitializer(cond_value->uniqueName());
    }
    const bool cond = static_cast<bool>(cond_tensor.data<int32_t>()[0]);
    auto &parent_graph = graph;
    const auto subgraph = if_node->g(cond ? kthen_branch : kelse_branch);

    std::unordered_map<std::string, Value *> unique_name_to_value_in_parent;

    for (auto *x : parent_graph.nodes()) {
      for (auto *x_output : x->outputs()) {
        unique_name_to_value_in_parent[x_output->uniqueName()] = x_output;
      }
    }
    std::unordered_map<std::string, Value *> value_dict;
    for (auto *node : subgraph->nodes()) {
      auto *new_node =
          parent_graph.create(node->kind(), node->outputs().size());
      new_node->insertBefore(if_node);
      new_node->copyAttributes(*node);
      for (const auto *input : node->inputs()) {
        const auto &unique_name = input->uniqueName();
        if (value_dict.find(unique_name) == value_dict.end()) {
          ONNX_ASSERT(input->node()->kind() == kCaptured);
          auto it = unique_name_to_value_in_parent.find(unique_name);
          if (it == unique_name_to_value_in_parent.end()) {
            // a value from the parent graph of parent_graph
            auto *captured_node = parent_graph.create(kCaptured, 1);
            captured_node->output()->setUniqueName(unique_name);
            new_node->addInput(captured_node->output());
          } else {
            new_node->addInput(it->second);
          }
        } else {
          new_node->addInput(value_dict[unique_name]);
        }
      }
      for (int i = 0; i < node->outputs().size(); i++) {
        const auto *output_in_subgraph = node->outputs()[i];
        auto *output_in_parent_graph = new_node->outputs()[i];
        value_dict[output_in_subgraph->uniqueName()] = output_in_parent_graph;
      }
    }
    const auto &subgraph_outputs = subgraph->outputs();
    for (int i = 0; i < subgraph_outputs.size(); i++) {
      auto *new_output = value_dict[subgraph_outputs[i]->uniqueName()];
      auto *if_output = if_node->outputs()[i];
      if_output->replaceAllUsesWith(new_output);
    }
    destroy_current = DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
