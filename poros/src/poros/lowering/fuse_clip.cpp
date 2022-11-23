/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_clip.cpp
 @author tianshaoqing@baidu.com
 @date 2022-08-01 16:08:26
 @brief
 */

#include "poros/lowering/fuse_clip.h"

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace baidu {
namespace mirana {
namespace poros {

/**
 * ReplaceClip
 * @param graph
 * @return true if graph changed, false if not
 */
bool FuseClip::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    // schema: aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    if (try_to_replace_clip(graph->block())) {
        std::string new_pattern = R"IR(
            graph(%x, %min, %max):
                %out : Tensor = aten::clamp(%x, %min, %max)
                return (%out))IR";

        std::string old_pattern = R"IR(
            graph(%x, %min, %max):
                %out : Tensor = aten::clip(%x, %min, %max)
                return (%out))IR";

        torch::jit::SubgraphRewriter std_rewriter;
        std_rewriter.RegisterRewritePattern(old_pattern, new_pattern);
        std_rewriter.runOnGraph(graph);

        return true;
    }
    return false;
}

/**
 * search for aten::clip recursively, record all findings
 * @param block
 * @return true if at least one aten::clip found, false if none found
 */
bool FuseClip::try_to_replace_clip(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //++it first, node may be destroyed later。
        for (auto sub_block: node->blocks()) {
            if (try_to_replace_clip(sub_block)) {
                graph_changed = true;
            }
        }
        //只处理 aten::clip场景
        if (node->kind() == torch::jit::aten::clip) {
            graph_changed = true;
            record_transform(torch::jit::aten::clip)->to(torch::jit::aten::clamp);
        }
    }
    return graph_changed;
}

FuseClip::FuseClip() = default;

REGISTER_OP_FUSER(FuseClip)

}
}
}// namespace