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
* @file eliminate_some_dict.cpp
* @author tianjinjin@baidu.com
* @date Wed Jan 26 19:41:32 CST 2022
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

struct EliminateSomeDict {
    EliminateSomeDict(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        GRAPH_DUMP("before eliminate_some_dicts Graph: ", graph_);
        bool changed = eliminate_dict_getitems(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after eliminate_some_dicts Graph: ", graph_);
        return;
    }

private:
    /*
    * @brief 
    * 将以下graph:
    *        %key1 : str = prim::Constant[value="first_key"]()
    *        %key2 : str = prim::Constant[value="second_key"]()
    *        %key3 : str = prim::Constant[value="third_key"]()
    *        %resdict = prim::DictConstruct(%key1, %value1, %key2, %value2, %key3, %value3)
    *        %res1 = aten::__getitem__(%resdict, %key1)
    *        %res2 = aten::__getitem__(%resdict, %key2)
    *        %1 = aten::matmul(%res1, %const.1)
    *        %2 = aten::matmul(%res2, %const.2)
    *        
    * 替换成以下graph：
    *        %1 = aten::matmul(%value1, %const.1)
    *        %2 = aten::matmul(%value2, %const.2)
    * 
    * 需要特别注意的是：
    *    1. 如果有多个__getitem__的时候，prim::DictConstruct不可删除，
    *       当所有的__getitem__都处理完了才能删除prim::DictConstruct
    * */
    bool is_dict_getitem_node(Node* node) {
        if (node->kind() != aten::__getitem__ ||
            node->inputs().at(1)->node()->kind() != prim::Constant) {
            return false;
        }
        auto producer_node = node->inputs().at(0)->node();
        if (producer_node->kind() != prim::DictConstruct ||
            producer_node->owningBlock() != node->owningBlock()) {
            return false;
        }

        //TODO: this can be changed to more loose condition
        for (auto &use: node->inputs().at(0)->uses()) {
            if (use.user->kind() != aten::__getitem__) {
                return false;
            }
        }

        return true;
    }

    bool eliminate_dict_getitem_after_construct(Node* node) {
        //提取get_item节点的key信息，即第二个参数的值。
        c10::optional<IValue> maybe_key = toIValue(node->inputs().at(1)->node()->output());
        if (!maybe_key.has_value()) {
            LOG(INFO) << "can not handle get_item node: " << node_info(node);
            return false;
        }
        auto key = maybe_key.value();

        //找到DictConstruct相应的key, 替换成相应的value。
        auto producer_node = node->inputs().at(0)->node();
        at::ArrayRef<Value*> producer_inputs = producer_node->inputs();
        size_t num_inputs = producer_inputs.size();
        for(size_t index = 0; index < num_inputs / 2; index++) {
            if (producer_inputs[index * 2]->node()->kind() != prim::Constant) {
                continue;
                // LOG(INFO) << "can not handle DictConstruct node: " << node_info(producer_node); 
                // return false;
            } else {
                c10::optional<IValue> ivalue = toIValue(producer_inputs[index * 2]->node()->output());
                if (ivalue.has_value() && ivalue.value() == key) {
                    //开启output value 替换大法。
                    node->outputs()[0]->replaceAllUsesWith(producer_inputs[index * 2 + 1]);
                    //本node可以destroy了。
                    LOG(INFO) << "replace all uses from value: %" << node->outputs()[0]->debugName() 
                            << " to value: %" << producer_inputs[index * 2 + 1]->debugName();
                    LOG(INFO) << "destroy getitem node now: " << node_info(node);
                    node->destroy();
                    break;
                }
            }
        }

        //当只有一个 getitem 节点的时候，producer 可以destroy了。(无需专门删除，EliminateDeadCode会处理掉producer_node)
        // if (producer_node->outputs()[0]->uses().size() == 0) {
        //     LOG(INFO) << "destroy dictConstruct node now: " << node_info(producer_node);
        //     producer_node->destroy();
        // }
        return true;
    }

    bool eliminate_dict_getitems(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // we might destroy the current node, so we need to pre-increment the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_dict_getitems(subblock);
            }
            if (is_dict_getitem_node(node)) {
                LOG(INFO) << "meet dict getitem after construct node :" << node_info(node);
                changed |= eliminate_dict_getitem_after_construct(node);
            }
        }
        return changed;        
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void eliminate_some_dict(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateSomeDict esd(std::move(graph));
    esd.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu