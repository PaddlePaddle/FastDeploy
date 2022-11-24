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
* @file link_mutable_list_pass.cpp
* @author tianshaoqing@baidu.com
* @date Thu May 9 11:15:49 CST 2022
* @brief
**/

#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "poros/context/poros_global.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;

struct LinkMutableList {
    LinkMutableList(std::shared_ptr<Graph> graph) : graph_(std::move(graph)), 
    _mutable_list_ops_set(PorosGlobalContext::instance().supported_mutable_ops_set){}

    void run() {    
        GRAPH_DUMP("Before linking mutable list Graph: ", graph_);
        bool changed = handle_mutable_list(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("After linking mutable list Graph: ", graph_);
        return;
    }

private:
    std::shared_ptr<Graph> graph_;

    const std::set<c10::Symbol> _mutable_list_ops_set;
    // handle_mutable_list功能：将mutable的op输入与输出串联起来

    // 通常（不含子block）情况:
    // ---------------------------------------------
    // %l1 : Tensor[] = aten::append(%list, %x1)
    // %l2 : Tensor[] = aten::append(%list, %x2)
    // %l3 : Tensor[] = aten::append(%list, %x3)
    // %l4 : Tensor[] = aten::append(%list, %x4)
    // ---------------------------------------------
    // 转化为以下IR:
    // ---------------------------------------------
    // %l1 : Tensor[] = aten::append(%list, %x1)
    // %l2 : Tensor[] = aten::append(%l1, %x2)
    // %l3 : Tensor[] = aten::append(%l2, %x3)
    // %l4 : Tensor[] = aten::append(%l3, %x4)
    // ---------------------------------------------

    // 特殊（含子block）情况，以下面的IR为例:
    // ----------------------------------------------
    // %list : Tensor[] = prim::ListConstruct(...)
    // %l1 : Tensor[] = aten::append(%list, %x1) 
    // %l2 : Tensor[] = aten::append(%list, %x2) 
    //  = prim::Loop(%5, %2)
    //   block0(%i : int):
    //     %l3 : Tensor[] = aten::_set_item(%list, %i, %x3) 
    //     %l4 : Tensor[] = aten::append(%list, %x4)
    //     -> (%2)
    // %l5 : Tensor[] = aten::append(%list, %x5) 
    // %%list2 : Tensor[] = aten::slice(%list, %4, %3, %4)
    // -----------------------------------------------
    // 只对最外层主图中的mutable list op 输入输出串起来，而子block中不串。转化为以下IR:
    // -----------------------------------------------
    // %list : Tensor[] = prim::ListConstruct(...)
    // %l1 : Tensor[] = aten::append(%list, %x1) 
    // %l2 : Tensor[] = aten::append(%l1, %x2) 
    //  = prim::Loop(%5, %2)
    //   block0(%i : int):
    //     %l3 : Tensor[] = aten::_set_item(%l2, %i, %x3) 
    //     %l4 : Tensor[] = aten::append(%l2, %x4)
    //     -> (%2)
    // %l5 : Tensor[] = aten::append(%l2, %x5)
    // %%list2 : Tensor[] = aten::slice(%l5, %4, %3, %4)
    // ----------------------------------------------
    // 只要保证子block中的mutable list op不合入子图就行，主图中子图的mutable可以不回传
    bool handle_mutable_list(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end(); ) {
            Node* node = *it;
            ++it;
            if (_mutable_list_ops_set.find(node->kind()) != _mutable_list_ops_set.end()) { 
                if (node->outputs().size() == 1) {
                    changed = true;
                    node->input(0)->replaceAllUsesAfterNodeWith(node, node->output(0)); 
                } else {
                    LOG(WARNING) << "mutable op: " << node_info(node) << " output size() != 1. " << 
                    "This situation is not yet supported.";
                }
            }
        }
        return changed;
    }
    // 以下是曾经实现的版本:
    // 版本一，本应是最理想的版本，但是无跑通
    // ----------------------------------------------
    // %list : Tensor[] = prim::ListConstruct(...)
    // %l1 : Tensor[] = aten::append(%list, %x1) 
    // %l2 : Tensor[] = aten::append(%l1, %x2) 
    //  = prim::Loop(%5, %2)
    //   block0(%i : int):
    //     %l3 : Tensor[] = aten::_set_item(%l2, %i, %x3)  <------执行到这步出错
    //     %l4 : Tensor[] = aten::append(%l2, %x4)
    //     -> (%2)
    // %l5 : Tensor[] = aten::append(%l2, %x5)
    // %%list2 : Tensor[] = aten::slice(%l2, %4, %3, %4)
    // -------------------------------------------------
    // *在子block中对某value使用replaceAllUsesAfterNodeWith时，如果block外面也有value的user的话jit会出错
    // 本例子中，由于l2在子block外部也有users，在给子block以外的%l2替换成%l3时会出错
    /*
    bool handle_mutable_list(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end(); ) {
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= handle_mutable_list(subblock);
            }
            if (_mutable_list_ops_set.find(node->kind()) != _mutable_list_ops_set.end()) {
                changed = true;
                node->input(0)->replaceAllUsesAfterNodeWith(node, node->output(0));
            }
        }
        return changed;
    }*/

    // 版本二，只替换同一block下且在本node之后的node（mutable）
    // 执行后:
    // ----------------------------------------------
    // %list : Tensor[] = prim::ListConstruct(...)
    // %l1 : Tensor[] = aten::append(%list, %x1) 
    // %l2 : Tensor[] = aten::append(%l1, %x2) 
    //  = prim::Loop(%5, %2)
    //   block0(%i : int):
    //     %l3 : Tensor[] = aten::_set_item(%list, %i, %x3) 
    //     %l4 : Tensor[] = aten::append(%l3, %x4)
    //     -> (%2)
    // %l5 : Tensor[] = aten::append(%l2, %x5)
    // %%list2 : Tensor[] = aten::slice(%l5, %4, %3, %4)
    // ------------------------------------------------
    // 作用域只能在自己node归属的block中，导致子block调用了最开始的mutable(%list)，
    // 主图中子图的mutable需要回传，子block中子图的mutable需要回传，依赖回传。
    /*
    bool handle_mutable_list(Block* block) {
        bool changed = false;
        for (auto it = block->nodes().begin(); it != block->nodes().end(); ) {
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= handle_mutable_list(subblock);
            }
            if (_mutable_list_ops_set.find(node->kind()) != _mutable_list_ops_set.end()) {
                changed = true;
                torch::jit::use_list use_list = node->input(0)->uses();
                for (size_t u = 0; u < use_list.size(); u++) {
                    // 只替换同一block下且在本node之后的node
                    if (use_list[u].user->owningBlock() == block && use_list[u].user->isAfter(node)) {
                        use_list[u].user->replaceInput(use_list[u].offset, node->output(0));
                    }
                }
            }
        }
        return changed;
    }*/
};

} // namespace

void link_mutable_list(std::shared_ptr<torch::jit::Graph> graph) {
    LinkMutableList lml(std::move(graph));
    lml.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu