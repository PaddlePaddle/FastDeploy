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

// Only support conv2d + bias now

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FusePaddleConvBias final : public PredicateBasedPass {
  explicit FusePaddleConvBias()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_paddle_conv_bias"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kAdd && node->inputs()[0]->node()->kind() == kConv &&
           node->inputs()[1]->node()->kind() == kConstant &&
           node->inputs()[0]->node()->inputs()[1]->node()->kind() == kConstant;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Conv is only used by Add
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }
    // check if bias is only used by Add
    if (n->inputs()[1]->uses().size() > 1) {
      return false;
    }

    Node* add = n;
    Node* conv = n->inputs()[0]->node();
    Node* bias = n->inputs()[1]->node();
    Node* weight = conv->inputs()[1]->node();

    if (conv->inputs().size() > 2) {
      return false;
    }

    Tensor bias_tensor = bias->t(kvalue);
    Tensor weight_tensor = weight->t(kvalue);
    const auto& bias_shape = bias_tensor.sizes();
    const auto& weight_shape = weight_tensor.sizes();
    if (bias_shape.size() != 4 || bias_shape.size() != weight_shape.size()) {
      return false;
    }
    if (bias_shape[0] != 1 || bias_shape[2] != 1 || bias_shape[3] != 1) {
      return false;
    }
    if (bias_shape[1] != weight_shape[0]) {
      return false;
    }
    // reshape bias node
    bias_tensor.sizes().clear();
    bias_tensor.sizes().push_back(weight_shape[0]);
    bias->t_(kvalue, std::move(bias_tensor));

    conv->addInput(bias->outputs()[0]);
    conv->output()->setSizes(add->output()->sizes());
    conv->output()->setElemType(add->output()->elemType());
    const bool replacing_success =
        tryReplacingAllUsesWith(add->output(), add->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
