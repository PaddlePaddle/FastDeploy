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
* @file segment.cpp
* @author tianjinjin@baidu.com
* @date Fri Mar 19 19:18:20 CST 2021
* @brief 
**/

#include "poros/compile/segment.h"

namespace baidu {
namespace mirana {
namespace poros {

std::vector<const torch::jit::Value*> sort_topological(const at::ArrayRef<const torch::jit::Value*> inputs,
                                                                const torch::jit::Block* cur_block,
                                                                bool reverse) {
    //if not in the same block. bypass it
    std::vector<const torch::jit::Value*> result;
    for (auto i : inputs) {
        if (i->node()->owningBlock() == cur_block) {
            result.push_back(i);
        }
    }

    if (reverse) {
        // Sort in reverse topological order
        std::sort(result.begin(), result.end(), [&](const torch::jit::Value* a, const torch::jit::Value* b) {
            return a->node()->isAfter(b->node());
        });
    } else {
        std::sort(result.begin(), result.end(), [&](const torch::jit::Value* a, const torch::jit::Value* b) {
            return a->node()->isBefore(b->node());
        });
    }
    return result;
}

std::vector<torch::jit::Value*> sort_topological (const at::ArrayRef<torch::jit::Value*> inputs,
                                        const torch::jit::Block* cur_block,
                                        bool reverse) {
    //if not in the same block. bypass it
    std::vector<torch::jit::Value*> result;
    for (auto i : inputs) {
        if (i->node()->owningBlock() == cur_block) {
            result.push_back(i);
        }
    }

    if (reverse) {
        // Sort in reverse topological order
        std::sort(result.begin(), result.end(), [&](torch::jit::Value* a, torch::jit::Value* b) {
            return a->node()->isAfter(b->node());
        });
    } else {
        std::sort(result.begin(), result.end(), [&](torch::jit::Value* a, torch::jit::Value* b) {
            return a->node()->isBefore(b->node());
        });
    }
    return result;
}

void stable_dfs(const torch::jit::Block& block, bool reverse,
               const std::vector<const torch::jit::Node*>& start,
               const std::function<bool(const torch::jit::Node*)>& enter,
               const std::function<bool(const torch::jit::Node*)>& leave)
{
    std::vector<NodeDFSResult> stack(start.size());
    for (size_t i = 0; i < start.size(); ++i) {
        stack[i] = NodeDFSResult{start[i], false};
    }
    
    std::unordered_map<const torch::jit::Node*, bool> visited;
    while(!stack.empty()) {
        NodeDFSResult w = stack.back();
        stack.pop_back();

        auto n = w.node;
        if (w.leave) {
            if (leave && !leave(n)) {
                return;
            }
            continue;
        }

        if (visited.find(n) != visited.end()) {
            continue;
        }
        visited[n] = true;

        if (enter && !enter(n)) {
            return;
        }

        if (leave) {
            stack.push_back(NodeDFSResult{n, true});
        }

        auto values = reverse ? n->inputs() : n->outputs();
        auto sorted_value_list = sort_topological(values, n->owningBlock(), false);
        for (auto value: sorted_value_list) {
            if (visited.find(value->node()) == visited.end()) {
                stack.push_back(NodeDFSResult{n, false});
            }
        }
    }
}

bool can_contract(const torch::jit::Node* from_node, 
                            const torch::jit::Node* to_node, 
                            const torch::jit::Block& block) {
    std::vector<const torch::jit::Node*> dfs_start_nodes;

    for (auto i: to_node->inputs()) {
        if (i->node() != from_node) {
            dfs_start_nodes.push_back(i->node());
        }
    }

    bool has_cycle = false;
    stable_dfs (block, /*reverse=*/true, dfs_start_nodes,  /*enter=*/nullptr,
            [&has_cycle, from_node](const torch::jit::Node* n) {
              if (n == from_node) {
                has_cycle = true;
                return false;
              }
              return true;
            });
    return !has_cycle;
}

torch::jit::Graph& get_subgraph(torch::jit::Node* n) {
    AT_ASSERT(n->kind() == torch::jit::prim::CudaFusionGroup);
    return *n->g(torch::jit::attr::Subgraph);
  }

torch::jit::Node* merge_node_into_subgraph(torch::jit::Node* group, torch::jit::Node* n) {
    auto& subgraph = get_subgraph(group);
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> inputs_map;
    size_t i = 0;
    size_t tensor_insert_idx = 0;
    //cache the original group input data
    AT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    for (auto input : group->inputs()) {
        inputs_map[input] = subgraph.inputs()[i++];
        if (input->type()->isSubtypeOf(c10::TensorType::get())) {
            tensor_insert_idx = i;
        }
    }

    torch::jit::WithInsertPoint guard(*subgraph.nodes().begin());
    for (auto input : n->inputs()) {
        //means we should add this new input
        if (inputs_map.count(input) == 0) {
            //consider tensortype first. (it's pytorch tradition)
            if (input->type()->isSubtypeOf(c10::TensorType::get())) {
                auto in_group = subgraph.insertInput(tensor_insert_idx);
                in_group->setType(input->type());
                inputs_map[input] = in_group;
                group->insertInput(tensor_insert_idx, input);
                tensor_insert_idx++;
            } else if ((input->type()->isSubtypeOf(c10::FloatType::get()) &&
                        input->node()->kind() != torch::jit::prim::Constant) ||
                        (n->kind() == torch::jit::aten::_grad_sum_to_size &&
                        input->type()->isSubtypeOf(c10::ListType::ofInts()))) {
                auto in_group = subgraph.addInput();
                in_group->setType(input->type());
                inputs_map[input] = in_group;
                group->addInput(input);
            } else if (input->node()->kind() == torch::jit::prim::Constant) {
                torch::jit::Node* in_const = subgraph.createClone(input->node(), [](torch::jit::Value*) -> torch::jit::Value* {
                    throw std::runtime_error("unexpected input");
                    });
                subgraph.insertNode(in_const);
                inputs_map[input] = in_const->output();
            } else {
                // TODO: we need to figure out what are supported input scalar
                LOG(WARNING) << "meet some unexpected node: " << input->node()->kind().toQualString();
                auto in_group = subgraph.addInput();
                in_group->setType(input->type());
                inputs_map[input] = in_group;
                group->addInput(input);
            }
        }
    }  // for (auto input : n->inputs())

    // copy n into the graph, remapping its inputs to internal nodes
    torch::jit::Node* in_graph = subgraph.createClone(
        n, [&](torch::jit::Value* k) -> torch::jit::Value* { return inputs_map[k]; });

    auto inputs = group->inputs();
    for (size_t i = 0; i < n->outputs().size(); ++i) {
        auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
        if (it != inputs.end()) {
            size_t p = it - inputs.begin();
            group->removeInput(p);
            subgraph.inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
            subgraph.eraseInput(p);
        }
    }
    return subgraph.insertNode(in_graph);
}

torch::jit::Node* change_node_to_subgraph(torch::jit::Node* group, torch::jit::Node* n)
{
    group->insertBefore(n);
    torch::jit::Node* mergedNode = merge_node_into_subgraph(group, n);
    get_subgraph(group).registerOutput(mergedNode->output());
    auto sel = group->addOutput();
    sel->copyMetadata(n->output());
    n->replaceAllUsesWith(group);
    n->destroy();
    return group;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
