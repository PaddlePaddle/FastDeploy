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
* @file eliminate_exception_pass.cpp
* @author tianjinjin@baidu.com
* @date Thu Sep 23 11:15:49 CST 2021
* @brief
**/
#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;
struct EliminateExceptionPasses {
    EliminateExceptionPasses(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        find_exception_if_node(graph_->block());
        torch::jit::EliminateDeadCode(graph_);
    }

private:
    bool is_exception_if_node(Node* n) {
        /// Check if this Node hosts a pattern like so:
        /// situation 1:
        ///  = prim::If(%5958)
        ///   block0():
        ///    -> ()
        ///   block1():
        ///     = prim::RaiseException(%45)
        ///    -> ()

        /// situation 2:
        ///  = prim::If(%5958)
        ///   block0():
        ///    = prim::RaiseException(%45)
        ///      -> ()
        ///   block1():
        ///   -> ()
        if (n->blocks().size() != 2) {
            return false;
        }
        auto arm1 = n->blocks()[0];
        auto arm2 = n->blocks()[1];
        if (arm1->outputs().size() != 0 || arm2->outputs().size() != 0) {
            // Make sure that the node doesn't actually produce any Value that are
            // used by other nodes
            return false;
        }

        auto arm1_start = arm1->nodes().begin();
        auto arm2_start = arm2->nodes().begin();

        if ((*arm1_start)->kind() == prim::Return) {
            // Make sure that block0 is solely the return
            if ((*arm2_start)->kind() != prim::RaiseException || (*(++arm2_start))->kind() != prim::Return) {
                // Make sure that block1 is solely just the exception and the return
                return false;
            }
            return true;
        }

        if ((*arm2_start)->kind() == prim::Return) {
            // Make sure that block1 is solely the return
            if ((*arm1_start)->kind() != prim::RaiseException || (*(++arm1_start))->kind() != prim::Return) {
                // Make sure that block0 is solely just the exception and the return
                return false;
            }
            return true;
        }
        return false;
    }

    void find_exception_if_node(Block* b) {
        for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
            auto n = *it;
            if (n->kind() == prim::If && is_exception_if_node(n)) {
                it.destroyCurrent();
            } else if (n->kind() == prim::If) {
                auto true_block = n->blocks()[0];
                find_exception_if_node(true_block);
                auto false_block = n->blocks()[1];
                find_exception_if_node(false_block);
            } else if (n->kind() == prim::Loop) {
                auto loop_block = n->blocks()[0];
                find_exception_if_node(loop_block);
            }
        }
    }

    std::shared_ptr<Graph> graph_;
};
} // namespace

void eliminate_exception_pass(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateExceptionPasses eppe(std::move(graph));
    eppe.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
