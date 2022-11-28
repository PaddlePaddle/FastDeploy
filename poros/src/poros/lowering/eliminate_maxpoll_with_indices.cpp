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
* @file eliminate_maxpoll_with_indices.cpp
* @author tianjinjin@baidu.com
* @date Tue Sep 13 11:06:07 CST 2022
* @brief
**/
#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;
/**
 * @brief 尝试用maxpool 代替 maxpool_with_indeces.
 * 以 maxpoll2d 为例：
 * maxpoll2d_with_indices 的schema为：aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
 * 而 maxpoll 的schema为：aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
 * 这两个op，输入参数完全一致，输出上，max_pool2d_with_indices有两个输出，第一个输出与max_pool2d的输出完全一致，第二个输出为indeces信息。
 * 当 max_pool2d_with_indices 的第二个输出indices，后续没有其他op使用该value的时候，
 * 我们直接用max_pool2d 替代 max_pool2d_with_indices。
 **/
struct EliminateMaxpollWithIndices {
    EliminateMaxpollWithIndices(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {    
        GRAPH_DUMP("before eliminate_maxpool_with_indices Graph: ", graph_);
        bool changed = eliminate_maxpool_with_indices(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after eliminate_maxpool_with_indices Graph: ", graph_);
        return;
    }

private:

    bool is_maxpoll_with_indices_pattern(Node* node) {
        if (node->kind() != aten::max_pool1d_with_indices &&
            node->kind() != aten::max_pool2d_with_indices &&
            node->kind() != aten::max_pool3d_with_indices){
            return false;
        }

        //当outputs的第二个值，也就是indices，没有被其他op使用的时候，满足替换条件。
        Value* indices = node->output(1);
        if (indices->uses().size() == 0) {
            return true;
        }
        return false;
    }

    bool replace_maxpool_with_indices(Node* node) {
        NodeKind replace_kind = aten::max_pool1d;
        if (node->kind() == aten::max_pool2d_with_indices) {
            replace_kind = aten::max_pool2d;
        } else if (node->kind() == aten::max_pool3d_with_indices) {
            replace_kind = aten::max_pool3d;
        };

        Node* maxpool_node = graph_->create(replace_kind, node->inputs());
        maxpool_node->output(0)->setType(node->output(0)->type());
        maxpool_node->copyMetadata(node);
        maxpool_node->insertBefore(node);
        node->output(0)->replaceAllUsesAfterNodeWith(node, maxpool_node->output(0));
        //node->output(0)->replaceAllUsesWith(maxpool_node->output(0));

        LOG(INFO) << "destroy maxpool_with_indeces node now: " << node_info(node);
        node->destroy();
        return true;
    }

    bool eliminate_maxpool_with_indices(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_maxpool_with_indices(subblock);
            }
            if (is_maxpoll_with_indices_pattern(node)) {
                LOG(INFO) << "find maxpoll with indices pattern: " << node_info(node);
                changed |= replace_maxpool_with_indices(node);
            }
        }
        return changed;
    }

std::shared_ptr<Graph> graph_;
};

} // namespace

void eliminate_maxpool_with_indices(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateMaxpollWithIndices emwi(std::move(graph));
    emwi.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
