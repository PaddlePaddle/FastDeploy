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
* @file replace_pad.cpp
* @author tianshaoqing@baidu.com
* @date 2022-11-09 19:34:40
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

struct ReplacePad {
    ReplacePad(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        GRAPH_DUMP("before replace pad Graph: ", graph_);
        bool changed = find_and_replace_pad(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after replace pad Graph: ", graph_);
        return;
    }

private:
    bool check_pad_constant_mode(Node* node) {
        bool is_constant_mode = false;
        // 检查schema是否为
        // aten::pad(Tensor self, int[] pad, str mode="constant", float? value=None) -> (Tensor)
        if (node->kind() == c10::Symbol::fromQualString("aten::pad") &&
            node->inputs().size() == 4 && 
            node->input(1)->type()->isSubtypeOf(c10::ListType::ofInts()) && 
            node->input(2)->type()->isSubtypeOf(c10::StringType::get())) {
            std::string pad_mode = toIValue(node->input(2)).value().toStringRef();
            if (pad_mode == "constant") {
                is_constant_mode = true;
            }
        }
        return is_constant_mode;
    }

    bool find_and_replace_pad(Block* block) {
        bool graph_changed = false;
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //++it first, node may be destroyed later。
            for (auto sub_block: node->blocks()) {
                if (find_and_replace_pad(sub_block)) {
                    graph_changed = true;
                }
            }
            // replace aten::pad with aten::constant_pad_nd when its padding mode is "constant".
            if (node->kind() == c10::Symbol::fromQualString("aten::pad") && 
                check_pad_constant_mode(node)) {
                torch::jit::Node* constant_pad_nd_node = graph_->create(torch::jit::aten::constant_pad_nd);
                constant_pad_nd_node->addInput(node->input(0));
                constant_pad_nd_node->addInput(node->input(1));
                constant_pad_nd_node->addInput(node->input(3));
                constant_pad_nd_node->insertBefore(node);
                node->output(0)->replaceAllUsesAfterNodeWith(node, constant_pad_nd_node->output(0));
                LOG(INFO) << "Replace aten::pad which padding mode is \"constant\" with aten::constant_pad_nd.";
                node->destroy();
                graph_changed = true;
            }
        }
        return graph_changed;
    }

    std::shared_ptr<Graph> graph_;
    std::unordered_set<c10::OperatorName> useless_schema_set_;
};

} // namespace

void replace_pad(std::shared_ptr<torch::jit::Graph> graph) {
    ReplacePad rp(std::move(graph));
    rp.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu