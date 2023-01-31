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
* @file remobe_simple_type_profile_nodes.cpp
* @author tianjinjin@baidu.com
* @date Mon May 10 11:06:53 CST 2021
* @brief 
**/
#include "poros/lowering/lowering_pass.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/jit_type.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;

struct RemoveSimpleTypeProfileNodes {
    RemoveSimpleTypeProfileNodes(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}
    void run() {
        remove_profile_nodes(graph_->block());
    }

private:
    bool profiled_with_different_types(Value* v) {
        std::vector<TypePtr> types;
        for (const auto& use : v->uses()) {
            if (use.user->kind() == prim::profile) {
                types.push_back(use.user->ty(attr::profiled_type));
            }
        }
        for (size_t i = 1; i < types.size(); ++i) {
            if (types.at(i - 1) != types.at(i)) {
                return true;
            }
        }
        return false;
    }

    bool is_simple_type_profile_node(Node* node) {
        return node->ty(attr::profiled_type) != TensorType::get();
    }

    void remove_profile_nodes(Block* block) {
        for (auto itr = block->nodes().begin(); itr != block->nodes().end(); itr++) {
            if (itr->kind() == prim::profile && is_simple_type_profile_node(*itr)) {  //todo
                itr->output()->replaceAllUsesWith(itr->input());
                if (!profiled_with_different_types(itr->input())) {
                    itr->input()->setType(itr->ty(attr::profiled_type));
                }
                itr.destroyCurrent();
            } else {
                for (Block* ib : itr->blocks()) {
                    remove_profile_nodes(ib);
                }
            }
        }
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void remove_simple_type_profile_nodes(std::shared_ptr<torch::jit::Graph> graph) {
    RemoveSimpleTypeProfileNodes(graph).run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
