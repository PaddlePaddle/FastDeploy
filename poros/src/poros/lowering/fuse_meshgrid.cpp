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
* @file fuse_meshgrid.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-04-29 14:56:48
* @brief
**/

#include "poros/lowering/fuse_meshgrid.h"

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace baidu {
namespace mirana {
namespace poros {
/**
 * rewrite meshgrid with `ones + transpose + mul`
 * @param graph
 * @return
 */
bool FuseMeshgrid::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    if (try_to_find_meshgrid(graph->block())) {
        std::string old_pattern = R"IR(
            graph(%x_1 : Tensor, %y_1 : Tensor ):
                %1 : Tensor[] = prim::ListConstruct(%x_1, %y_1)
                %2 : Tensor[] = aten::meshgrid(%1)
                return (%2))IR";

        std::string new_pattern = R"IR(
            graph(%x_1 : Tensor, %y_1 : Tensor):
                %device.1 : Device = prim::device(%x_1)
                %2 : NoneType = prim::Constant()
                %3 : int = prim::Constant[value=1]()
                %4 : int = prim::Constant[value=0]()
                %5 : int[] = aten::size(%y_1)
                %6 : int = aten::__getitem__(%5, %4)
                %7 : int[] = prim::ListConstruct(%3, %6)
                %x_dtype : int = prim::dtype(%x_1)
                %8 : Tensor = aten::ones(%7, %x_dtype, %2, %device.1, %2)
                %10 : Tensor = aten::unsqueeze(%x_1, %4)
                %11 : Tensor = aten::transpose(%10, %4, %3)
                %12 : Tensor = aten::mul(%8, %11)

                %25 : int[] = aten::size(%x_1)
                %26 : int = aten::__getitem__(%25, %4)
                %27 : int[] = prim::ListConstruct(%26, %3)
                %y_dtype : int = prim::dtype(%y_1)
                %28 : Tensor = aten::ones(%27, %y_dtype, %2, %device.1, %2)
                %29 : Tensor = aten::unsqueeze(%y_1, %4)
                %18 : Tensor = aten::mul(%28, %29)

                %19 : Tensor[] =  prim::ListConstruct(%12, %18)
                return (%19))IR";
        torch::jit::SubgraphRewriter std_rewriter;
        std_rewriter.RegisterRewritePattern(old_pattern, new_pattern);
        std_rewriter.runOnGraph(graph);

        return true;
    }
    return false;

}

FuseMeshgrid::FuseMeshgrid() = default;

/**
 * find out whether meshgrid exists
 * @param block
 * @return bool: true if meshgrid exists, else false
 */
bool FuseMeshgrid::try_to_find_meshgrid(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //++it first, node may be destroyed later。
        for (auto sub_block: node->blocks()) {
            if (try_to_find_meshgrid(sub_block)) {
                graph_changed = true;
            }
        }
        //只处理 aten::conv + batch_norm的场景
        if (node->kind() != torch::jit::aten::meshgrid) {
            continue;
        }

        record_transform(torch::jit::aten::meshgrid)->to(torch::jit::aten::ones, torch::jit::aten::unsqueeze,
                                                         torch::jit::aten::transpose, torch::jit::aten::mul);
        graph_changed = true;
    }
    return graph_changed;
}

REGISTER_OP_FUSER(FuseMeshgrid)

}
}
}// namespace
