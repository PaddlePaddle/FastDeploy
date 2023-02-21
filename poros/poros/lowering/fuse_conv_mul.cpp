// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
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

/**
* @file: fuse_conv_mul2.cpp
* @author: zhangfan51@baidu.com
* @data: 2022-04-24 18:43:02
* @brief: 
**/ 
#include "poros/lowering/fuse_conv_mul.h"

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

using namespace torch::jit;

FuseConvMul::FuseConvMul() = default;

/**
 * FuseConvMul
 * @param graph
 * @return true if graph changed, false if not
 */
bool FuseConvMul::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    graph_ = graph;
    return try_to_fuse_conv_mul(graph_->block());
}

/**
 * search for aten::conv + aten::mul patten for fuse
 * @param block
 * @return true if fuse success
 */
bool FuseConvMul::try_to_fuse_conv_mul(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //先++it, node 可能destroy掉。
        for (auto sub_block : node->blocks()) {
            if (try_to_fuse_conv_mul(sub_block)) {
                graph_changed = true;
            }
        }
        // find the op by "aten::mul".Scalar(Tensor self, Scalar other) -> Tensor"
        if (node->kind() != aten::mul) {
            continue;
        }

        // find "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor"
        if (!node->inputs()[0]->type()->isSubtypeOf(c10::TensorType::get()) ||
            node->inputs()[1]->node()->kind() != prim::Constant) {
            continue;
        }

        auto conv = node->inputs()[0]->node();
        if((conv->kind() != aten::conv2d && conv->kind() != aten::conv1d && conv->kind() != aten::conv3d) ||
            node->inputs()[0]->uses().size() != 1) {
            continue;
        }

        if (!(conv->inputs()[1])->type()->isSubtypeOf(c10::TensorType::get()) || // conv_weight
            conv->inputs()[1]->uses().size() != 1) {
            continue;
        }
        at::Tensor conv_w = toIValue(conv->inputs()[1])->toTensor();
        float scale = toIValue(node->inputs()[1])->toScalar().toFloat();
            
        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        // check bias
        if (conv->inputs()[2]->type()->isSubtypeOf(c10::TensorType::get())) {
            if (conv->inputs()[2]->uses().size() != 1) {
                continue;
            }
            at::Tensor conv_b = toIValue(conv->inputs()[2])->toTensor();
            auto new_conv_b = graph_->insertConstant(conv_b * scale);
            new_conv_b->setDebugName(conv->inputs()[2]->debugName() + ".scale");
            // 替换conv的bias值
            conv->inputs().at(2)->replaceAllUsesWith(new_conv_b);
        }

        auto new_conv_w = graph_->insertConstant(conv_w * scale);
        new_conv_w->setDebugName(conv->inputs()[1]->debugName() + ".scale");

        // 替换conv的weight值
        conv->inputs().at(1)->replaceAllUsesWith(new_conv_w);
        // 把所有的aten::mul的output的users更改为conv的output
        node->output()->replaceAllUsesWith(conv->output());

        LOG(INFO) << "Found fuse_conv2d_mul, node = " << *node;
        // 删除 aten::mul节点
        node->removeAllInputs();
        node->destroy();
        graph_changed = true;
    }
    return graph_changed;
}

REGISTER_OP_FUSER(FuseConvMul)

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu