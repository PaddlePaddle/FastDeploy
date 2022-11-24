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
* @file poros_util.h
* @author tianjinjin@baidu.com
* @date Wed Apr  7 17:52:36 CST 2021
* @brief 
**/

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <torch/script.h>

namespace baidu {
namespace mirana {
namespace poros {

//以下两个函数输出单个节点的信息
inline std::string node_info_with_attr(const torch::jit::Node *n) {
    std::stringstream ss;
    n->print(ss, 0, {}, /**print_source_locations = */false,
        /* print_attributes = */true,
        /* print_scopes = */ false,
        /* print_body = */ false);
    return ss.str();
}

inline std::string node_info(const torch::jit::Node *n) {
    std::stringstream ss;
    n->print(ss, 0, {}, /**print_source_locations = */false,
        /* print_attributes = */false,
        /* print_scopes = */ false,
        /* print_body = */ false);
    return ss.str();
}

//when a node info like: %1126 : Float(*, 2, strides=[2, 1], requires_grad=0, device=cuda:0) = aten::squeeze(%output1.1, %self.new_length.1075)
//we set the output like: [aten::squeeze][ouput:%1126][input:%output1.1, %self.new_length.1075]
inline std::string layer_info(const torch::jit::Node *n) {
    std::stringstream ss;
    ss << "[" << n->kind().toQualString() << "]";
    auto outs = n->outputs();
    if (outs.size() > 0) {
        ss << "[output:";
        size_t i = 0;
        for (auto out : outs) {
            if (i++ > 0) {
                ss << ",";
            }      
            ss << "%" << out->debugName();
        }
        ss << "]";
    }
    auto ins = n->inputs();
    if (outs.size() == 0 && ins.size() != 0) {
        ss << "[][input:";
        size_t i = 0;
        for (auto in : ins) {
            if (i++ > 0) {
                ss << ",";
            }      
            ss << "%" << in->debugName();
        }
        ss << "]";
    }
    return ss.str();
}

// inline std::string node_info(const torch::jit::Node* n) {
//     std::stringstream ss;
//     ss << *n;
//     std::string node_info = ss.str();
//     node_info.erase(std::remove(node_info.begin(), node_info.end(), '\n'), node_info.end());
//     return node_info;
// }

int merge_graph_to_module(std::shared_ptr<torch::jit::Graph>& to_merge_graph, 
                            torch::jit::Module& module,
                            bool init_module_ptr);

torch::jit::Module build_tmp_module(std::shared_ptr<torch::jit::Graph>& sub_graph);


bool gen_dims_for_tensor(const torch::jit::Value* value, std::vector<int64_t>& dims);

/**
 * @brief update global context when some Value copy happened in the Segment progress && engine transform progress.
 *        当在子图分割阶段或者后续engine转换阶段出现value的复制场景时，调用改函数，完成必要的value全局信息的拷贝。
 *        当前（2022.01）主要实现 value_dynamic_shape_map 中，value的shape信息的拷贝。
 * @param [in] old_value : 原value
 * @param [in] new_value : 新的value，新value的meta信息从原value拷贝而来。
 * @return null
 **/
void update_global_context(torch::jit::Value* old_value, torch::jit::Value* new_value);

/**
 * @brief update global context when some Value copy happened in the Segment progress && engine transform progress.
 *        当子图分割中出现node融合时，需要更新node维度的key
 * @param [in] old_node : 原node
 * @param [in] new_node : 新的node，新value的meta信息从原value拷贝而来。
 * @return null
 **/
void update_global_list_size_map_node_key_context(torch::jit::Node* old_node, torch::jit::Node* new_node);

/**
 * @brief update global context when some Value copy happened in the Segment progress && engine transform progress.
 *        当子图分割中出现node融合时，需要更新node维度的key
 * @param [in] old_node : 原node
 * @param [in] new_node : 新的node，新value的meta信息从原value拷贝而来。
 * @return null
 **/
void update_global_int_intlist_map_context(torch::jit::Value* old_value, torch::jit::Value* new_value);

/**
 * @brief unmerge the subgraph to its parent graph（especially when engine transform failed）
 *        把子图的节点信息，重新merge的父图里面去（尤其是在子图转engine失败需要fallback的场景）
 *
 * @param [in] subgraph_node : 类型为CudaFusionGroup的特殊node。
 * @return null
 **/
void unmerge_subgraph(torch::jit::Node* subgraph_node);

/**
 * @brief 在输入block及其子block中遍历CudaFusionGroup节点，放入to_optimized_nodes中准备优化。
 *
 * @param [in] block : 需要遍历CudaFusionGroup节点的block。
 * @param [out] to_optimized_nodes : 输出遍历到的CudaFusionGroup节点集合。
 * @return null
 **/
void find_to_optimized_nodes(torch::jit::Block* block, std::vector<torch::jit::Node*>& to_optimized_nodes);


/********************************************************************
             SOME DEPRECATED FUNCTIONS BELOW
*********************************************************************/
//DEPRECATED
bool gen_dims_for_scarlar(const torch::jit::Value* value, std::vector<int64_t>& dims);
//DEPRECATED
bool gen_dims_for_tensorlist(const torch::jit::Value* value, std::vector<int64_t>& dims);

//判断某个节点是否是可以不展开的liner节点，否则的话会展开很多的分支，把整个graph切分的过于细碎。
//DEPRECATED
bool is_linear_if_node(torch::jit::Node* node);

//DEPRECATED
std::vector<torch::jit::Value*> extract_linear_input(torch::jit::Node *node);

//判断某个if节点的输入，是否是一个aten::dim 和 const 比较生成的，如果是的，这个if节点依赖的判断条件很可能是一个常量。
//DEPRECATED
bool is_dim_equal_if_node(torch::jit::Node* node);

//当if的条件恒成立时，将if条件去掉，相应的block提出来。
//DEPRECATED
void inline_if_body(torch::jit::Block* body);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
