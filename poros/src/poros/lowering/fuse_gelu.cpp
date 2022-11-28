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
* @file fuse_gelu.cpp
* @author tianshaoqing@baidu.com
* @date 2022-10-20 14:39:32
* @brief
**/

#include "poros/lowering/fuse_gelu.h"

#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/version.h>

namespace baidu {
namespace mirana {
namespace poros {
/**
 * Rewrite aten::gelu to the fast version:
 * y = 0.5 * x * (1 + tanh(sqrt(2 / Pi) * (x + 0.044715 * x^3)))
 * Note: This may result in a small diff.
 * @param graph
 * @return 
 */
bool FuseGelu::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    if (try_to_find_gelu(graph->block())) {
        std::string gelu_pattern;
        std::string gelu_reduce_pattern;

        if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 12) {
            gelu_pattern = R"IR(
                graph(%x):
                    %out : Tensor = aten::gelu(%x)
                    return (%out))IR";

            gelu_reduce_pattern = R"IR(
                graph(%x.1 : Tensor):
                    %1 : float = prim::Constant[value=0.044714999999999998]()
                    %2 : float = prim::Constant[value=0.79788456080000003]()
                    %3 : int = prim::Constant[value=3]()
                    %4 : float = prim::Constant[value=1.0]()
                    %5 : float = prim::Constant[value=0.5]()
                    %6 : Tensor = aten::pow(%x.1, %3)
                    %7 : Tensor = aten::mul(%6, %1)
                    %8 : Tensor = aten::add(%7, %x.1, %4)
                    %9 : Tensor = aten::mul(%8, %2)
                    %10 : Tensor = aten::tanh(%9)
                    %11 : Tensor = aten::add(%10, %4, %4)
                    %12 : Tensor = aten::mul(%11, %x.1)
                    %13 : Tensor = aten::mul(%12, %5)
                    return (%13))IR";
        } else {
            gelu_pattern = R"IR(
                graph(%x : Tensor, %approximate : str):
                    %out : Tensor = aten::gelu(%x, %approximate)
                    return (%out))IR";

            gelu_reduce_pattern = R"IR(
                graph(%x.1 : Tensor, %approximate):
                    %1 : float = prim::Constant[value=0.044714999999999998]()
                    %2 : float = prim::Constant[value=0.79788456080000003]()
                    %3 : int = prim::Constant[value=3]()
                    %4 : float = prim::Constant[value=1.0]()
                    %5 : float = prim::Constant[value=0.5]()
                    %6 : Tensor = aten::pow(%x.1, %3)
                    %7 : Tensor = aten::mul(%6, %1)
                    %8 : Tensor = aten::add(%7, %x.1, %4)
                    %9 : Tensor = aten::mul(%8, %2)
                    %10 : Tensor = aten::tanh(%9)
                    %11 : Tensor = aten::add(%10, %4, %4)
                    %12 : Tensor = aten::mul(%11, %x.1)
                    %13 : Tensor = aten::mul(%12, %5)
                    return (%13))IR";
        }
        torch::jit::SubgraphRewriter gelu_rewriter;
        gelu_rewriter.RegisterRewritePattern(gelu_pattern, gelu_reduce_pattern);
        gelu_rewriter.runOnGraph(graph);

        return true;
    }
    return false;
}

FuseGelu::FuseGelu() = default;

/**
 * find out whether gelu exists.
 * @param block
 * @return bool: true if aten::gelu exists, else false.
 */
bool FuseGelu::try_to_find_gelu(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //++it first, node may be destroyed later。
        for (auto sub_block: node->blocks()) {
            if (try_to_find_gelu(sub_block)) {
                graph_changed = true;
            }
        }
        
        if (node->kind() == torch::jit::aten::gelu) { 
            record_transform(torch::jit::aten::gelu)->to(torch::jit::aten::pow, torch::jit::aten::mul,
                                                            torch::jit::aten::add, torch::jit::aten::tanh);
            graph_changed = true;
        }
    }
    return graph_changed;
}

REGISTER_OP_FUSER(FuseGelu)

}
}
}// namespace