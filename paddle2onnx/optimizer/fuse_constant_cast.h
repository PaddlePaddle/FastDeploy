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

struct FuseConstantCast final : public PredicateBasedPass {
  explicit FuseConstantCast()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_constant_cast"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kCast &&
           node->inputs()[0]->node()->kind() == kConstant;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* cast = n;
    Node* constant = n->inputs()[0]->node();
    Tensor t = constant->t(kvalue);
    auto dtype = cast->i(kto);
    t.elem_type() = dtype;
    constant->t_(kvalue, std::move(t));
    if (!tryReplacingAllUsesWith(cast->output(), cast->inputs()[0])) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
