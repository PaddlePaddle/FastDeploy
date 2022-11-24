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
* @file segment.h
* @author tianjinjin@baidu.com
* @date Thu Mar 18 14:33:54 CST 2021
* @brief 
**/

#pragma once

#include <string>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <torch/script.h>

#include "poros/engine/iengine.h"

namespace baidu {
namespace mirana {
namespace poros {

struct NodeDFSResult {
    const torch::jit::Node* node;
    bool leave;  // Are we entering or leaving n?
};

//template <typename T>  //should I?
class NodeLink {
public:
    NodeLink(torch::jit::Node* n)
      : _size(1), _head(nullptr), _value(n) {}

    ~NodeLink() {
        if (_head != nullptr) {
            _head = nullptr;
        }
        if (_value != nullptr) {
            _value = nullptr;
        }
    }

    int size() {return find_root()->_size; }
    int merge(NodeLink* other) {
        NodeLink* a = find_root();
        NodeLink* b = other->find_root();
        if (a == b) {
            return 0;
        }
        b->_head = a;
        a->_size += b->_size;
        return 0;
    };

    // Retrieves the value for the root of the set.
    torch::jit::Node* head_value() { return find_root()->_value; }

    // Returns the value for the object.
    torch::jit::Node* value() const { return _value; }
    //int64_t value_index() {return _value->topo_position_; }

private:
    NodeLink* find_root() {
        if (!_head) {
            return this;
        }
        _head = _head->find_root();
        return _head;
    };
    
    int _size;
    NodeLink* _head;
    torch::jit::Node* _value;
};


struct SegmentOptions {
    int minimum_segment_size = 2;  //每个setment至少包含多少个node。
};

struct Segment {
    Segment() {}
    Segment(std::set<torch::jit::Node*>& nodes)
                   : nodes(nodes){}
    std::set<torch::jit::Node*> nodes;
};

using SegmentVector = std::vector<Segment>;
using ValueVector = std::vector<const torch::jit::Value*>;

ValueVector sort_topological (const at::ArrayRef<const torch::jit::Value*> inputs,
                                        const torch::jit::Block* cur_block,
                                        bool reverse = false);
std::vector<torch::jit::Value*> sort_topological (const at::ArrayRef<torch::jit::Value*> inputs,
                                        const torch::jit::Block* cur_block,
                                        bool reverse = false);

void stable_dfs(const torch::jit::Block& block, bool reverse,
               const std::vector<const torch::jit::Node*>& start,
               const std::function<bool(const torch::jit::Node*)>& enter,
               const std::function<bool(const torch::jit::Node*)>& leave);


bool can_contract(const torch::jit::Node* from_node, 
                            const torch::jit::Node* to_node, 
                            const torch::jit::Block& block);

torch::jit::Graph& get_subgraph(torch::jit::Node* n);
torch::jit::Node* merge_node_into_subgraph(torch::jit::Node* group, torch::jit::Node* n);
torch::jit::Node* change_node_to_subgraph(torch::jit::Node* group, torch::jit::Node* n);

//void segment_graph_new(std::shared_ptr<torch::jit::Graph>& graph, IEngine* engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
