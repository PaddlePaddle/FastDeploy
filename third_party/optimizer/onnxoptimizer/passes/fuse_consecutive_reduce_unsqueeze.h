/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_proto_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> reduction_operators{
    kReduceL1,   kReduceL2,  kReduceLogSum, kReduceLogSumExp, kReduceMax,
    kReduceMean, kReduceMin, kReduceProd,   kReduceSum,       kReduceSumSquare};

struct FuseConsecutiveReduceUnsqueeze final : public PredicateBasedPass {
  explicit FuseConsecutiveReduceUnsqueeze()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_reduce_unsqueeze";
  }
  static bool IsAxesAnAttr(const Graph &graph, const Node *n) {
    int opset_version = getOpsetVersion(graph);
    int opset_threshold;
    if (n->kind() == kUnsqueeze || n->kind() == kReduceSum) {
      opset_threshold = 12;
      return opset_version <= opset_threshold && opset_version != 0;
    }
    return true;
  }
  static bool getAxes(const Node *n, Graph &graph, std::vector<int64_t> &axes) {
    if (IsAxesAnAttr(graph, n)) {
      if (!n->hasAttribute(kaxes)) {
        return false;
      }
      axes = n->is(kaxes);
    } else {
      if (n->inputs().size() < 2) {
        return false;
      }
      auto axes_value = n->inputs()[1];
      if ((axes_value->node()->kind() != kConstant &&
           axes_value->node()->kind() != kParam)) {
        return false;
      }
      Tensor axes_t;
      if (axes_value->node()->kind() == kConstant) {
        axes_t = axes_value->node()->t(kvalue);
      } else {
        const auto axes_i = graph.getInitializer(axes_value->uniqueName());
        axes_t = *axes_i;
      }
      axes = ParseData<int64_t>(&axes_t);
    }
    return true;
  }
  bool patternMatchPredicate(Node *node) override {
    // check that the current node is of type Unsqueeze and has defined axes
    bool cur_node_check = node->kind() == kUnsqueeze;
    if (cur_node_check) {
      Node *prev_node = node->inputs()[0]->node();
      // check that the previous node a reduction operator and has defined
      // axes/keepdims
      bool reduction_node_check = reduction_operators.find(prev_node->kind()) !=
                                      reduction_operators.end() &&
                                  prev_node->hasAttribute(kkeepdims);
      if (reduction_node_check) {
        // insure that keepdims is set to false currently
        return prev_node->i(kkeepdims) == 0;
      }
    }
    return false;
  }
  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    Node *prev_node = node->inputs()[0]->node();
    std::vector<int64_t> axes;
    bool success = getAxes(node, graph, axes);
    if (!success) {
      return false;
    }
    std::vector<int64_t> prev_axes;
    success = getAxes(prev_node, graph, prev_axes);
    if (!success) {
      return false;
    }
    if (axes != prev_axes) {
      return false;
    }
    Node *reduction_op = node->inputs()[0]->node();
    // set keepdims flag to be true
    reduction_op->i_(kkeepdims, 1);
    // remove unnecessary unsqueeze
    reduction_op->output()->setSizes(node->output()->sizes());
    reduction_op->output()->setElemType(node->output()->elemType());
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
