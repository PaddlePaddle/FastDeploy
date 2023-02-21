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
* @file replace_illegal_constant.cpp
* @author tianshaoqing@baidu.com
* @date 2022-06-01 19:34:40
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

struct ReplaceIllegalConstant {
    ReplaceIllegalConstant(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        GRAPH_DUMP("before replace_illegal_constant Graph: ", graph_);
        bool changed = find_and_replace_illegal_constant(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after replace_illegal_constant Graph: ", graph_);
        return;
    }

private:
    bool check_constant_is_illegal(Node* node) {
        bool is_illegal = false;
        // 检查constant输出是否是int
        if (node->kind() == torch::jit::prim::Constant && node->outputs().size() > 0 && 
            node->output(0)->type()->kind() == c10::TypeKind::IntType) {
            torch::jit::IValue const_value = toIValue(node->output(0));
            // 这里的toInt返回的是int64_t
            long const_double = const_value.toInt();
            // 判断int是否等于INT64_MAX，且有users
            if ((const_double == INT64_MAX) && node->output(0)->hasUses()) {
                is_illegal = true;
                auto const_node_users = node->output(0)->uses();
                // 判断逻辑，目前只遇到了slice end输入为非法constant的情况，其他情况遇到再加
                for (size_t u = 0; u < const_node_users.size(); u++) {
                    if (const_node_users[u].user->kind() != torch::jit::aten::slice) {
                        is_illegal = false;
                        break;
                    }
                }
            }
        }
        return is_illegal;
    }

    bool find_and_replace_illegal_constant(Block* block) {
        bool graph_changed = false;
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //++it first, node may be destroyed later。
            for (auto sub_block: node->blocks()) {
                if (find_and_replace_illegal_constant(sub_block)) {
                    graph_changed = true;
                }
            }
            
            if (node->kind() == torch::jit::prim::Constant && 
                check_constant_is_illegal(node)) {
                // 将slice end输入替换为none
                torch::jit::Node* none_node = graph_->createNone();
                none_node->insertBefore(node);
                node->output(0)->replaceAllUsesAfterNodeWith(node, none_node->output(0));
                LOG(INFO) << "Found illegal constant INT64_MAX used as index by aten::slice. Replace it with Constant None.";
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

void replace_illegal_constant(std::shared_ptr<torch::jit::Graph> graph) {
    ReplaceIllegalConstant ric(std::move(graph));
    ric.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu