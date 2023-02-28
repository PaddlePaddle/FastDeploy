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
* @file: try_to_freeze_array.cpp
* @author: zhangfan51@baidu.com
* @data: 2022-03-23 15:53:29
* @brief: 
**/ 
#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/passes/constant_propagation.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;
/**
    * @brief 尝试展开for循环内list.append()的情况
    * try to expand
    * graph(%feat.29 : Tensor):
    *     %256 : int = prim::Constant[value=2]()
    *     %257 : float = prim::Constant[value=2.]()
    *     %scale_factors.88 : float[] = prim::ListConstruct()
    *         = prim::Loop(%256, %2146)
    *             block0(%746 : int):
    *                 %747 : float[] = aten::append(%scale_factors.88, %257)
    *             -> (%2146)
    * as
    * graph(%feat.29 : Tensor):
    *     %256 : int = prim::Constant[value=2]()
    *     %257 : float = prim::Constant[value=2.]()
    *     %scale_factors.88 : float[] = prim::ListConstruct()
    *     %747 : float[] = aten::append(%scale_factors.88, %257)
    *     %748 : float[] = aten::append(%scale_factors.88, %257)
    **/ 


struct FreeezeListConstruct {
    FreeezeListConstruct(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        try_to_freeze_list_construct(graph_->block());
        // 运行一遍常量折叠
        torch::jit::ConstantPropagation(graph_);
    }

private:
    template<typename T>
    void replace_contant_list_construct(Node* node, std::vector<T> &data_array) {
        LOG(INFO) << "try to replace the output of node :" << node_info(node)
                << " with constant value " << data_array;
        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        auto list_const = graph_->insertConstant(data_array);
        node->outputs().at(0)->replaceAllUsesWith(list_const);
    }

    void try_to_freeze_list_construct(Block* block) {
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //先++it, node 可能destroy掉。
            for (auto block : node->blocks()) {
                try_to_freeze_list_construct(block);
            }

            // 找到 prim::ListConstruct
            if (node->kind() != prim::ListConstruct) {
                continue;
            }
            // 判断 ListConstruct的output为float[] or int[]
            if (!(node->outputs()[0])->type()->isSubtypeOf(c10::ListType::ofFloats()) && 
                !(node->outputs()[0])->type()->isSubtypeOf(c10::ListType::ofInts())) {
                continue;
            }
            // 判断 ListConstruct的inputs应为空，否则会有值不相同
            if (node->inputs().size() != 0) {
                continue;
            }
            //only do for float[] and int[]
            // 判断该ListConstruct的所有使用者，是否仅做过一次aten::append修改
            int use_flag = 0;
            Node* app_node = nullptr;
            for (auto &use : node->outputs()[0]->uses()) {
                if (use.user->owningBlock() != node->owningBlock() && 
                    use.user->kind() == aten::append) {
                    use_flag++;
                    app_node = use.user;
                }
            }
            if (use_flag != 1) {
                continue;
            }
            // 判断append的block 是放在prim::Loop中的, 且在该Loop里只有1个block
            Block* app_block = app_node->owningBlock();
            Node* loop_node = app_block->owningNode();
            // 目前先仅考虑owingNode为prim::Loop的情况，如后面遇到其他类似pattern再做相应判断调整
            if (loop_node->kind() != prim::Loop || loop_node->blocks().size() > 1) {
                continue;
            }

            auto app_it = app_block->nodes().begin();
            std::vector<Node*> app_block_nodes;
            while (app_it != app_block->nodes().end()) {
                app_block_nodes.push_back(*app_it);
                ++app_it;
            }
            // 仅处理形如这种的情况：
            //         block0(%746 : int):
            //             %747 : float[] = aten::append(%scale_factors.88, %257)
            //         -> (%2146)
            // block 中仅包括1个append
            if (app_block_nodes.size() != 1) {
                LOG(INFO) << "freeze_list_construct: append block nodes size is more than 1. ";
                continue;
            }
            // prim::Loop的 循环次数必须为prim::Constant, append的value也必须为prim::Constant.
            if ((loop_node->inputs()[0])->node()->kind() != prim::Constant ||
                (app_node->inputs()[1])->node()->kind() != prim::Constant ) {
                LOG(INFO) << "freeze_list_construct: append's input or loop's input type is not prim::Constant.";
                continue;
            }
            auto loop_max = toIValue(loop_node->inputs()[0]->node()->output()).value().toInt();
            auto loop_cond = toIValue(loop_node->inputs()[1]->node()->output()).value().toBool();
            // loop_cond must be true here, check again.
            if (!loop_cond) {
                continue;
            }
            auto value = toIValue((app_node->inputs()[1])->node()->output()).value();
            if (value.isInt()) {
                std::vector<int64_t> array_value(loop_max, value.toInt());
                replace_contant_list_construct(node, array_value);
            } else if (value.isDouble()) {
                std::vector<double> array_value(loop_max, value.toDouble()); 
                replace_contant_list_construct(node, array_value);
            } else {
                continue;
            }
            //destroy app_block下所有node
            for (size_t i = 0; i < app_block_nodes.size(); ++i) {
                app_block_nodes[i]->destroy();
            }
            // loop_node->destroy();
        }
    }
    std::shared_ptr<Graph> graph_;
};

} // namespace

void freeze_list_construct(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros freeze_list_construct passes";
    FreeezeListConstruct flc(std::move(graph));
    flc.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu