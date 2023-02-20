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
* @file eliminate_some_list.cpp
* @author tianjinjin@baidu.com
* @date Thu Sep 23 11:15:49 CST 2021
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

struct EliminateSomeList {
    EliminateSomeList(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {    
        GRAPH_DUMP("before eliminate_some_lists Graph: ", graph_);
        bool changed = eliminate_list_unpacks(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        changed = eliminate_list_getitems(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after eliminate_some_lists Graph: ", graph_);
        return;
    }

private:
    /*
    * @brief 
    * 将以下graph：
    *        %reslist = prim::ListConstruct(%1, %2, %3)
    *        %res1, %res2, %res3 = prim::ListUnpack(%reslist)
    *        %4 = aten::matmul(%res1, %const.0)
    *        %5 = aten::matmul(%res2, %const.1)
    *        %6 = aten::matmul(%res3, %const.2)
    *
    * 或者以下graph：
    *        %reslist = prim::ListConstruct(%1)
    *        %reslist.2 = aten::append(%reslist, %2)
    *        %reslist.3 = aten::append(%reslist, %3)
    *        %res1, %res2, %res3 = prim::ListUnpack(%reslist)
    *        %4 = aten::matmul(%res1, %const.0)
    *        %5 = aten::matmul(%res2, %const.1)
    *        %6 = aten::matmul(%res3, %const.2)
    *        
    * 替换成以下graph：
    *        %4 = aten::matmul(%1, %const.0)
    *        %5 = aten::matmul(%2, %const.1)
    *        %6 = aten::matmul(%3, %const.1)
    * 
    * 需要特别注意的是：
    *    1. 如果是ListConstruct + append 的模式，这些节点都得在同一个block，否则如果append在subblock下面，难以确定append的次数。
    *    2. ListConstruct 和 ListUnpack 间，除了同block下的append之外，不应该有其他可能改变该list的算子出现，否则会带来非预期的影响。
    */
    bool is_list_unpack_pattern(Node* node) {
        if (node->kind() != prim::ListUnpack) {
            return false;
        }

        auto input_value = node->inputs().at(0);
        auto producer_node = node->inputs().at(0)->node();

        if (producer_node->kind() != prim::ListConstruct ||
            producer_node->owningBlock() != node->owningBlock()) {
            return false;
        }
        
        for (auto &use: input_value->uses()) {
            if (use.user->kind() == aten::append) {
                if (use.user->output()->uses().size() != 0 ||
                use.user->owningBlock() != node->owningBlock()) {
                    //aten::apend 的output有被其他节点用，或者不在一个block
                    LOG(INFO) << "find unmatched pattern: " << node_info(node);
                    return false;                   
                }
            } else if (use.user->kind() == prim::ListUnpack) {
                continue;
            } else {
                //不满足我们找寻的条件
                LOG(INFO) << "find unmatched pattern: " << node_info(node);
                return false;
            }
        }
        return true;
    }

    bool eliminate_list_unpack_after_construct(Node* node) {
        //auto input_value = node->inputs().at(0);
        auto producer_node = node->inputs().at(0)->node();
        
        auto output_num = node->outputs().size();
        auto input_num = producer_node->inputs().size() + node->inputs()[0]->uses().size() - 1;
        if (input_num != output_num) {
            LOG(WARNING) << "ListConstruct + aten::append input_num not equal prim::ListUnpack output_num, "
                        << "bypass this node: " << node_info(node);
            return false;
        }
            
        std::vector<Value*> input_value_list;
        //prim::ListConstruct 的input 倒腾进去
        for (auto value : producer_node->inputs()) {
            input_value_list.push_back(value);
        }

        //TODO: 是否要排个序？？
        std::vector<Node*> append_node_list;
        for (auto &use: node->inputs()[0]->uses()) {
            if (use.user->kind() == aten::append) {
                //aten::append 的 第二个 input 倒腾进去
                input_value_list.push_back(use.user->inputs()[1]);
                append_node_list.push_back(use.user);
            }
        }

        if (input_value_list.size() != output_num) {
            LOG(WARNING) << "ListConstruct + aten::append input_num not equal prim::ListUnpack output_num, "
                        << "bypass this node: " << node_info(node);
            return false;
        }
            
        int index = 0;
        //开启output value 替换大法。
        for (auto output_value : node->outputs()) {
            auto replace_value = input_value_list[index++];
            output_value->replaceAllUsesWith(replace_value);
        }

        //本node可以destroy了。
        LOG(INFO) << "destroy listUnpack node now: " << node_info(node);
        node->destroy();

        //aten::append可以destroy了。
        for (auto &append_node: append_node_list) {
            LOG(INFO) << "destroy aten::append node now: " << node_info(append_node);
            append_node->destroy();            
        }
            
        //producer_node可以destroy了。
        LOG(INFO) << "destroy listConstruct node now: " << node_info(producer_node);
        producer_node->destroy();
        return true;
    }

    /*
    * @brief 
    * 将以下graph：
    *        %reslist = prim::ListConstruct(%1, %2, %3)
    *        %4 : int = prim::Constant[value=-1]()
    *        %res1 = aten::__getitem__(%reslist, %const.0)
    *        %5 = aten::matmul(%res1, %const.1)
    *
    * 或者以下graph：
    *        %reslist = prim::ListConstruct(%1)
    *        %reslist.2 = aten::append(%reslist, %2)
    *        %reslist.3 = aten::append(%reslist, %3)
    *        %4 : int = prim::Constant[value=-1]()
    *        %res1 = aten::__getitem__(%reslist, %4)
    *        %5 = aten::matmul(%res1, %const.1)
    *        
    * 替换成以下graph：
    *        %5 = aten::matmul(%3, %const.1)
    * 
    * 需要特别注意的是：
    *    1. 如果有多个__getitem__的时候，append信息和listconstruct不能够删除，
    *       当所有的__getitem__都处理完了才能删除append和listconstruct
    * */
    bool is_list_getitem_node(Node* node) {
        if (node->kind() != aten::__getitem__ ||
            node->inputs().at(1)->node()->kind() != prim::Constant) {
            return false;
        }

        auto producer_node = node->inputs().at(0)->node();
        if (producer_node->kind() != prim::ListConstruct ||
            producer_node->owningBlock() != node->owningBlock()) {
            return false;
        }

        return true;
    }

    bool eliminate_list_getitem_after_construct(Node* node) {
        auto input_value = node->inputs().at(0);
        auto producer_node = input_value->node();

        int get_item_count = 0;
        for (auto &use: input_value->uses()) {
            if (use.user->kind() == aten::append) {
                if (use.user->output()->uses().size() != 0 ||
                use.user->owningBlock() != node->owningBlock()) {
                    //aten::apend 的output有被其他节点用，或者不在一个block
                    LOG(INFO) << "find unmatched pattern: " << node_info(node);
                    return false;           
                }
            } else if (use.user->kind() == aten::__getitem__) {
                get_item_count++;
                continue;
            } else {
                //不满足我们找寻的条件
                LOG(INFO) << "find unmatched pattern: " << node_info(node);
                return false;
            }
        }

        LOG(INFO) << "find list getitem after construct pattern: " << node_info(node);
        auto input_num = producer_node->inputs().size() + node->inputs()[0]->uses().size() - 1;

        std::vector<Value*> input_value_list;
        //prim::ListConstruct 的input 倒腾进去
        for (auto value : producer_node->inputs()) {
            input_value_list.push_back(value);
        }

        //TODO: 是否要排个序？？
        std::vector<Node*> append_node_list;
        for (auto &use: node->inputs()[0]->uses()) {
            if (use.user->kind() == aten::append) {
                //aten::append 的 第二个 input 倒腾进去
                input_value_list.push_back(use.user->inputs()[1]);
                append_node_list.push_back(use.user);
            }
        }

        //求取index的值。
        int64_t index = toIValue((node->inputs()[1])->node()->output()).value().toInt();
        index = index < 0 ? input_num + index : index;
        LOG(INFO) << "calculate getitem index number is : " << index;

        //开启output value 替换大法。
        node->outputs()[0]->replaceAllUsesWith(input_value_list[index]);

        //本node可以destroy了。
        LOG(INFO) << "destroy getitem node now: " << node_info(node);
        node->destroy();

        //当只有一个 getitem 节点的时候，aten::append 和 producer 都可以destroy了。
        if (get_item_count == 1) {
            for (auto &append_node : append_node_list) {
                LOG(INFO) << "destroy aten::append node now: " << node_info(append_node);
                append_node->destroy();
            }
            /* 以下的迭代方式可能出core。
            auto users_count = producer_node->output()->uses().size();
            for (int user_index = users_count; user_index >= 0; user_index--) {
                auto append_node = producer_node->output()->uses()[user_index].user;
                if (append_node->kind() == aten::append) {
                    LOG(INFO) << "destroy aten::append node now: " << node_info(append_node);
                    append_node->destroy();
                }
            }*/
            //producer_node可以destroy了。
            LOG(INFO) << "destroy listConstruct node now: " << node_info(producer_node);
            producer_node->destroy();
        }
        return true;
    }

    bool eliminate_list_unpacks(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_list_unpacks(subblock);
            }
            if (is_list_unpack_pattern(node)) {
                LOG(INFO) << "find list unpack after construct pattern: " << node_info(node);
                changed |= eliminate_list_unpack_after_construct(node);
            }
        }
        return changed;
    }

    bool eliminate_list_getitems(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= eliminate_list_getitems(subblock);
            }
            if (is_list_getitem_node(node)) {
                //LOG(INFO) << "meet list getitem after construct node :" << node_info(node);
                changed |= eliminate_list_getitem_after_construct(node);
            }
        }
        return changed;        
    }

std::shared_ptr<Graph> graph_;
};

} // namespace

void eliminate_some_list(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateSomeList esl(std::move(graph));
    esl.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
