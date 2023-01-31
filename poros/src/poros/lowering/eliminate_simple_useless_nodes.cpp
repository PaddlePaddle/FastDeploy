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
* @file eliminate_simple_useless_nodes.cpp
* @author tianshaoqing@baidu.com
* @date 2022-08-25 11:06:26
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

struct EliminateSimpleUselessNodes {
    EliminateSimpleUselessNodes(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
        useless_schema_set_.emplace(torch::jit::parseSchema("aten::dropout(Tensor input, float p, "
        "bool train) -> Tensor").operator_name());
        useless_schema_set_.emplace(torch::jit::parseSchema("aten::warn(str message, int stacklevel=2) "
        "-> ()").operator_name());
    }

    void run() {
        GRAPH_DUMP("before eliminate_simple_useless_nodes Graph: ", graph_);
        bool changed = find_and_eliminate_simple_useless_nodes(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after eliminate_simple_useless_nodes Graph: ", graph_);
        return;
    }

private:
    bool find_and_eliminate_simple_useless_nodes(Block* block) {
        bool graph_changed = false;
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //++it first, node may be destroyed later。
            for (auto sub_block: node->blocks()) {
                if (find_and_eliminate_simple_useless_nodes(sub_block)) {
                    graph_changed = true;
                }
            }
            
            if (node->maybeSchema() && useless_schema_set_.count(node->schema().operator_name())) {
                if (node->kind() == torch::jit::aten::warn) {
                    node->destroy();
                }
                if (node->kind() == torch::jit::aten::dropout) {
                    node->output(0)->replaceAllUsesWith(node->input(0));
                    node->destroy();
                }
                graph_changed = true;
            }
        }
        return graph_changed;
    }

    std::shared_ptr<Graph> graph_;
    std::unordered_set<c10::OperatorName> useless_schema_set_;
};

} // namespace

void eliminate_simple_useless_nodes(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateSimpleUselessNodes esun(std::move(graph));
    esun.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu