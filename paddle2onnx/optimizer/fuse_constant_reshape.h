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
//   B = Reshape(Constant)
// After:
//   B = Constant (Constant with new shape)

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConstantReshape final : public PredicateBasedPass {
  explicit FuseConstantReshape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_constant_reshape"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kReshape &&
           node->inputs()[0]->node()->kind() == kConstant;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Constant is only used by Reshape
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* reshape = n;
    Node* constant = n->inputs()[0]->node();

    // Process 'reshape' data
    std::vector<int64_t> shape;
    if (reshape->hasAttribute(kshape)) {
      // opset 5 and below
      shape = reshape->is(kshape);
    } else {
      // opset 6 and above - first check if 'reshape' has 'shape' input
      // constant
      if (reshape->inputs()[1]->node()->kind() != kConstant) {
        return false;
      }
      if (reshape->inputs()[1]->uses().size() > 1) {
        return false;
      }
      Node* shape_const = reshape->inputs()[1]->node();
      Tensor t = shape_const->t(kvalue);
      shape = ParseData<int64_t>(&t);
    }

    int allow_zero = 0;
    Symbol sym = Symbol("allowzero");
    if (reshape->hasAttribute(sym)) {
      allow_zero = reshape->i(sym);
    }

    Tensor t = constant->t(kvalue);
    const auto& ori_size = t.sizes();

    // process 0 in shape
    if (allow_zero != 0) {
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
          // illegal situation
          if (ori_size.size() <= i) {
            return false;
          }
          shape[i] = ori_size[i];
        }
      }
    }

    // process -1 in shape
    int count_of_unkown = 0;
    int index_of_unkown = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == -1) {
        count_of_unkown += 1;
        index_of_unkown = i;
      }
    }
    // illegal situtaion
    if (count_of_unkown > 1) {
      return false;
    }
    int64_t numel = std::accumulate(ori_size.begin(), ori_size.end(), 1,
                                    std::multiplies<int>());
    if (index_of_unkown >= 0) {
      int64_t value_of_unkown = -1 * numel /
                                std::accumulate(shape.begin(), shape.end(), 1,
                                                std::multiplies<int>());
      shape[index_of_unkown] = value_of_unkown;
    }

    t.sizes().clear();
    t.sizes().insert(t.sizes().begin(), shape.begin(),
                     shape.begin() + shape.size());
    constant->t_(kvalue, std::move(t));

    // update constant node
    constant->output()->setSizes(reshape->output()->sizes());
    constant->output()->setElemType(reshape->output()->elemType());
    const bool replacing_success =
        tryReplacingAllUsesWith(reshape->output(), reshape->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
