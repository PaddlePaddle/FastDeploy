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
* @file fuse_hard_swish.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-04-07 15:30:35
* @brief
**/
#include "poros/lowering/fuse_hard_swish.h"

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace baidu {
namespace mirana {
namespace poros {

/**
 * FuseHardSwish
 * @param graph
 * @return true if graph changed, false if not
 */
bool FuseHardSwish::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    if (try_to_find_hardswish(graph->block())) {
        std::string new_pattern = R"IR(
            graph(%x):
                %1 : int = prim::Constant[value=1]()
                %3 : int = prim::Constant[value=3]()
                %6 : int = prim::Constant[value=6]()
                %7 : int = prim::Constant[value=0]()
                %x_1 : Tensor = aten::add(%x, %3, %1)
                %x_2 : Tensor = aten::clamp(%x_1, %7, %6)
                %x_3 : Tensor = aten::mul(%x, %x_2)
                %out : Tensor = aten::div(%x_3, %6)
                return (%out))IR";

        std::string old_pattern = R"IR(
            graph(%x):
                %out: Tensor = aten::hardswish(%x)
                return (%out))IR";

        torch::jit::SubgraphRewriter std_rewriter;
        std_rewriter.RegisterRewritePattern(old_pattern, new_pattern);
        std_rewriter.runOnGraph(graph);

        return true;
    }

    return false;
}

/**
 * search for hardswish activation recursively, record all findings
 * @param block
 * @return true if at least one hardswish found, false if none found
 */
bool FuseHardSwish::try_to_find_hardswish(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //++it first, node may be destroyed later。
        for (auto sub_block: node->blocks()) {
            if (try_to_find_hardswish(sub_block)) {
                graph_changed = true;
            }
        }
        //只处理 aten::hardswish的场景
        if (node->kind() != torch::jit::aten::hardswish) {
            continue;
        }
        record_transform(torch::jit::aten::hardswish)->to(torch::jit::aten::add, torch::jit::aten::clamp, torch::jit::aten::div);

        graph_changed = true;
    }
    return graph_changed;
}

FuseHardSwish::FuseHardSwish() = default;

REGISTER_OP_FUSER(FuseHardSwish)

}
}
}// namespace