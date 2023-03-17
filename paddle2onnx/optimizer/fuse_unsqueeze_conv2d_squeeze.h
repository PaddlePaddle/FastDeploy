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
//   X = Tensor(N, C, H)
//   B = Unsqueeze(X, 2)
//   C = Conv2d(B)
//   D = Squeeze(C, 2)
// After:
//   D = Conv1d(X)

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseUnsqueezeConv2dSqueeze final : public PredicateBasedPass {
  explicit FuseUnsqueezeConv2dSqueeze()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_unsqueeze_conv2d_squeeze";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kSqueeze &&
           node->inputs()[0]->node()->kind() == kConv &&
           node->inputs()[0]->node()->inputs()[0]->node()->kind() == kUnsqueeze;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* squeeze_node = n;
    Node* conv_node = n->inputs()[0]->node();
    Node* unsqueeze_node = conv_node->inputs()[0]->node();
    if (squeeze_node->inputs()[0]->uses().size() > 1) {
      return false;
    }
    if (conv_node->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* weight_node = conv_node->inputs()[1]->node();
    if (weight_node->kind() != kConstant) {
      return false;
    }
    Tensor weight = weight_node->t(kvalue);
    if (weight.sizes().size() != 4) {
      return false;
    }
    if (weight.sizes()[2] != 1) {
      return false;
    }
    {
      std::vector<int64_t> axes;
      if (squeeze_node->hasAttribute(kaxes)) {
        // opset 12 and below
        axes = squeeze_node->is(kaxes);
      } else {
        // opset 13 and above
        if (squeeze_node->inputs()[1]->node()->kind() != kConstant) {
          return false;
        }
        if (squeeze_node->inputs()[1]->uses().size() > 1) {
          return false;
        }
        Tensor t = squeeze_node->inputs()[1]->node()->t(kvalue);
        axes = ParseData<int64_t>(&t);
      }
      if (axes.size() != 1 || axes[0] != 2) {
        return false;
      }
    }
    {
      std::vector<int64_t> axes;
      if (unsqueeze_node->hasAttribute(kaxes)) {
        // opset 12 and below
        axes = unsqueeze_node->is(kaxes);
      } else {
        // opset 13 and above
        if (unsqueeze_node->inputs()[1]->node()->kind() != kConstant) {
          return false;
        }
        if (unsqueeze_node->inputs()[1]->uses().size() > 1) {
          return false;
        }
        Tensor t = unsqueeze_node->inputs()[1]->node()->t(kvalue);
        axes = ParseData<int64_t>(&t);
      }
      if (axes.size() != 1 || axes[0] != 2) {
        return false;
      }
    }
    // update conv weight
    weight.sizes().erase(weight.sizes().begin() + 2);
    weight_node->t_(kvalue, std::move(weight));

    if (conv_node->hasAttribute(kdilations)) {
      std::vector<int64_t> dilations = conv_node->is(kdilations);
      if (dilations.size() != 2 || dilations[0] != 1) {
        return false;
      }
      dilations.erase(dilations.begin() + 0);
      conv_node->is_(kdilations, std::move(dilations));
    }
    if (conv_node->hasAttribute(kkernel_shape)) {
      std::vector<int64_t> kernel_shape = conv_node->is(kkernel_shape);
      if (kernel_shape.size() != 2 || kernel_shape[0] != 1) {
        return false;
      }
      kernel_shape.erase(kernel_shape.begin() + 0);
      conv_node->is_(kkernel_shape, std::move(kernel_shape));
    }
    if (conv_node->hasAttribute(kpads)) {
      std::vector<int64_t> pads = conv_node->is(kpads);
      if (pads.size() != 4 || pads[0] != 0 || pads[2] != 0) {
        return false;
      }
      pads.erase(pads.begin() + 2);
      pads.erase(pads.begin() + 0);
      conv_node->is_(kpads, std::move(pads));
    }
    if (conv_node->hasAttribute(kstrides)) {
      std::vector<int64_t> strides = conv_node->is(kstrides);
      if (strides.size() != 2 || strides[0] != 1) {
        return false;
      }
      strides.erase(strides.begin() + 0);
      conv_node->is_(kstrides, std::move(strides));
    }

    conv_node->replaceInput(0, unsqueeze_node->inputs()[0]);
    if (!tryReplacingAllUsesWith(unsqueeze_node->output(),
                                 unsqueeze_node->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(squeeze_node->output(),
                                 squeeze_node->inputs()[0])) {
      return false;
    }
    //    unsqueeze_node->destroy();
    //    squeeze_node->destroy();
    //    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
