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
* @file partition.cpp
* @author tianjinjin@baidu.com
* @date Thu Jun  3 15:10:30 CST 2021
* @brief 
**/

#include "poros/compile/partition.h"

namespace baidu {
namespace mirana {
namespace poros {

// utility function to check if the node implies broadcast on a given shape (
// assumed to be shape of an input tensor)
// limitations:
//   1. we rely on shape information to judge this. so we would require output
//      shape to be available;
//   2. we basically compares given shape to the shape of the only output of
//      the node and return true if it implies broadcast from the former to the
//      latter.
bool maybeBroadcastOnShape(
    const torch::jit::Node* node,
    const std::vector<c10::optional<int64_t>>& shape) {
    //TODO: add outputs size check
    //TORCH_INTERNAL_ASSERT(n->outputs().size() == 1, "not expecting multiple outputs from a node, graph partitioning logic needs to be updated");
    // assumes that if output is not a tensor type, it's not broadcasting
    if (auto out_type = node->output(0)->type()->cast<c10::TensorType>()) {
        if (out_type->dim()) {
            if (out_type->dim().value() < shape.size()) {
                // no broadcast for reduction operation;
                return false;
            } else if (out_type->dim().value() > shape.size()) {
                // increased rank means there is reduction;
                return true;
            } else {
                // same rank, we need to iterate through sizes and check if size-1
                // exists in input `shape`
                for (const auto& opt_size : shape) {
                // TODO: not sure if we need to check for output size != 1, since we
                // are currently marking all size-1 dimension as broadcast in codegen.
                    if (opt_size.has_value() && opt_size.value() == 1) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
};

// bool hasReductionOperation(const torch::jit::Node* node) {
// if (torch::jit::fuser::cuda::isReductionNode(node)) {
//     return true;
// }
// if (node->kind() == torch::jit::prim::CudaFusionGroup) {
//     for (auto n : node->g(torch::jit::attr::Subgraph)->nodes()) {
//         if (hasReductionOperation(n)) {
//             return true;
//         }
//     }
// }
// return false;
// }
    
bool createTrickyBroadcast(const torch::jit::Node* consumer, const torch::jit::Node* producer) {
    
    auto count_broadcasting_in_node =
        [](const torch::jit::Node* node,
        const std::vector<c10::optional<int64_t>>& shape,
        size_t offset) {
            int num_broadcasting = 0;
            if (node->kind() == torch::jit::prim::CudaFusionGroup) {
                // be careful here as `subgraph_input`, as its name suggests, is in a
                // different fraph from `node`.
                const auto& subgraph_input =node->g(torch::jit::attr::Subgraph)->inputs()[offset];
                for (const auto& use : subgraph_input->uses()) {
                    if (maybeBroadcastOnShape(use.user, shape)) {
                        num_broadcasting++;
                    }
                }
            } else {
                if (maybeBroadcastOnShape(node, shape)) {
                    num_broadcasting++;
                }
            }
            return num_broadcasting;
        };
        
    // case 1. We check shared inputs to `producer` & `consumer`;
    for (int i = 0; i < static_cast<int>(producer->inputs().size()); i++) {
        auto n_input = producer->input(i);
        auto n_input_type = n_input->type()->cast<c10::TensorType>();
        if (n_input_type != nullptr && n_input_type->sizes().sizes()) {
            std::vector<c10::optional<int64_t>> n_input_shape = n_input_type->sizes().sizes().value();
            int num_broadcasting = 0;
            
            // check broadcasting for the n_input inside `consumer`;
            for (const auto& use : n_input->uses()) {
                if (use.user == consumer) {
                    num_broadcasting += count_broadcasting_in_node(consumer, n_input_shape, use.offset);
                }
            }

            // if no broadcasting happened for consumer, there's no point check
            // multiple broadcasting in producer alone;
            if (num_broadcasting == 0) {
                continue;
            }

            // check broadcasting for n_input inside `producer`;
            num_broadcasting += count_broadcasting_in_node(producer, n_input_shape, i);

            // encounted multiple broadcasting scheme for a single TV, we will not be
            // able to schedule this, prevent the fusion; (case 1)
            if (num_broadcasting > 1) {
                return true;
            }
        }
    }

    // case 2. We check input to `consumer` that is also the output from
    // `producer`
    for (int i = 0; i < static_cast<int>(producer->outputs().size()); i++) {
        auto n_output = producer->output(i);
        auto n_output_type = n_output->type()->cast<c10::TensorType>();
        if (n_output_type != nullptr && n_output_type->sizes().sizes()) {
            std::vector<c10::optional<int64_t>> n_output_shape = n_output_type->sizes().sizes().value();
            int num_broadcasting = 0;
            // If we only look at case 1 & case 2, we need to check broadcast of
            // `n_output` inside `producer`, if it is a `prim::CudaFusionGroup`.
            // this is actually not necessary when we consider case 3, as we avoid
            // broadcasting on outputs already;

            // TODO: merge this code with case 1.
            // check broadcasting for the n_output inside `consumer`;
            bool use_as_output = false;
            for (const auto& use : n_output->uses()) {
                if (use.user == consumer) {
                    num_broadcasting += count_broadcasting_in_node(consumer, n_output_shape, use.offset);
                } else {
                    // case 3. output is used by other nodes not the consumer, no
                    //         broadcasting is allowed;
                    use_as_output = true;
                }
            }

            // encounted multiple broadcasting scheme for a single TV, we will not be
            // able to schedule this, prevent the fusion; (case 2)
            // Alternatively, if use_as_output is true, we would not permit broadcast
            // at all. (case 3)
            if (num_broadcasting > (use_as_output ? 0 : 1)) {
                return true;
            }
        }
    }
    return false;
}

bool is_node_fusable(const torch::jit::Node* node, IEngine* engine) {
    if (node->kind() == torch::jit::prim::CudaFusionGroup || (engine->is_node_supported(node))) {
        return true;
    }
    return false;
}

bool is_node_fusable(const torch::jit::Node* fusion, 
                    const torch::jit::Node* node,
                    IEngine* engine) {
    if (is_node_fusable(node, engine) && !createTrickyBroadcast(fusion, node)) {
        return true;
    }
    return false;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
