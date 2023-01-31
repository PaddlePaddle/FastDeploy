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
* @file eliminate_useless_copy.cpp
* @author tianjinjin@baidu.com
* @date Thu Dec 16 16:27:02 CST 2021
* @brief
**/
#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/ir/ir.h>
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

struct EliminateUselessCopy {
    EliminateUselessCopy(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        GRAPH_DUMP("before eliminate_useless_copys Graph: ", graph_);
        bool changed = eliminate_useless_copys(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after eliminate_useless_copys Graph: ", graph_);
        return;
    }

private:
    /*
    * @brief 
    * 相关schema: aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
    * pytorch原始实现: https://github.com/pytorch/pytorch/blob/v1.9.0/aten/src/ATen/native/Copy.cpp#L246
    * 
    * 针对 %output = aten::copy_(%self, %src, %non_blocking) 形式的node。
    * 找出符合以下条件的aten::copy_
    *    1. %output 没有被其他node使用
    *    2. %self 除了aten::copy_ 本node外，没有被其他node使用
    *    当一个op同时满足以上两个条件时，认为该node可以直接删除。
    */
    bool is_node_useless_copy_pattern(Node* node) {
        if (node->kind() != aten::copy_) {
            return false;
        }

        if (node->inputs().at(0)->uses().size() == 1 &&
            node->outputs().at(0)->uses().size() == 0) {
            return true;
        }

        LOG(INFO) << "find unmatched pattern: " << node_info(node);
        return false;
    }

    bool eliminate_useless_copy_node(Node* node) {
        //本node可以destroy了。
        LOG(INFO) << "destroy aten::copy_ node now: " << node_info(node);
        node->destroy();
        return true;
    }

    bool eliminate_useless_copys(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_useless_copys(subblock);
            }
            if (is_node_useless_copy_pattern(node)) {
                LOG(INFO) << "find useless aten copy pattern: " << node_info(node);
                changed |= eliminate_useless_copy_node(node);
            }
        }
        return changed;
    }

std::shared_ptr<Graph> graph_;
};

} // namespace

void eliminate_useless_copy(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateUselessCopy euc(std::move(graph));
    euc.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu