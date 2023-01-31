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
* @file: fuse_copy.cpp
* @author: tianjinjin@baidu.com
* @data: Wed Jun 16 20:28:36 CST 2021
* @brief: 
**/ 

#include "poros/lowering/fuse_copy.h"

#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/version.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

using namespace torch::jit;

FuseCopy::FuseCopy() = default;

/**
 * FuseCopy
 * @param graph
 * @return true if graph changed, false if not
 */
bool FuseCopy::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    graph_ = graph;
    GRAPH_DUMP("before fuse copy ops Graph: ", graph_);
    bool fused = try_to_fuse_copy(graph_->block());
    if (fused) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
            //EraseNumberTypesOnBlock(graph_->block);
            //EliminateDeadCode(graph_->block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
    }
    GRAPH_DUMP("after fuse copy ops Graph: ", graph_);
    return fused;
}

/**
* @brief 创建aten::size节点，用于生成给定dim的size信息
* **/
Value* FuseCopy::create_size_of_dim(Value* input, int64_t dim, Node* insertBefore) {
    auto graph = input->owningGraph();
    WithInsertPoint guard(insertBefore);
    auto size = graph->insert(aten::size, {input, dim});
    LOG(INFO) << "create_size_of_dim before node: " << node_info(insertBefore);
    LOG(INFO) << "create aten::size node: " << node_info(size->node());
    return size;
}

/**
* @brief 对value进行处理，尤其是需要对维度进行补充的情况（也就是被select给降维的情况需要把对应的维度补回来）。
* **/
void FuseCopy::adjust_value(Graph* graph,
                    Node* index_put_node,
                    const std::vector<Node*>& slice_and_select_nodes,
                    Value* orig_data) {
    //获取常量value的rank信息，如果value的rank为0或者为1，则不需要专门处理(虽然也可以在这里处理...)
    //如果rank不是0，则这个tensor可能是select 生成的，需要提升维度，使得跟self维度一致后，再broadcast。
    bool need_unsqueeze_value = true;
    Value* value =  index_put_node->inputs().at(2);
    if (value->node()->kind() == prim::Constant) {
        at::Tensor value_tensor = toIValue(value).value().toTensor();
        int64_t value_rank = value_tensor.dim();
        if (value_rank == 0 || value_rank == 1) {
            need_unsqueeze_value = false;
        }
    }

    if (need_unsqueeze_value == true) {
        int64_t dim_offset = 0;
        for (auto it = slice_and_select_nodes.rbegin(); it != slice_and_select_nodes.rend(); ++it) {
            Node* node = *it;
            int64_t dim = toIValue(node->inputs().at(1)).value().toInt();
            if (dim < 0) {
                std::shared_ptr<c10::TensorType> input_type = orig_data->type()->expect<TensorType>();
                if (input_type->dim().has_value()) {
                    int64_t rank = static_cast<int64_t>(input_type->dim().value());
                    dim = dim + rank - dim_offset;
                }
            }
            dim = dim + dim_offset;

            if (node->kind() == aten::select) {
                //需要对value进行维度的还原。
                WithInsertPoint guard(index_put_node);
                Value* unsqueeze = graph->insert(aten::unsqueeze, {index_put_node->inputs().at(2), dim});
                LOG(INFO) << "create aten::unsqueeze node: " << node_info(unsqueeze->node());
                index_put_node->replaceInput(2, unsqueeze);
                dim_offset++;
            }
        }
    }
    return;
}

/**
* @brief 创建aten::tensor节点包装indices信息。
* **/
Value* FuseCopy::convert_select_to_index(Value* index, Node* insertBefore) {
    // Create index tensor based on index input of aten::select node.
    auto graph = insertBefore->owningGraph();
    WithInsertPoint guard(insertBefore);
    Node* indices = graph->create(aten::tensor, {
        index,
        graph->insertConstant(c10::ScalarType::Long),
        //graph->insertConstant(torch::Device(torch::DeviceType::CUDA, 0)),
        graph->insertConstant(torch::Device(at::kCPU)),
        graph->insertConstant(false)});

    indices->copyMetadata(insertBefore);
    indices->insertBefore(insertBefore);
    LOG(INFO) << "convert_select_to_index before node: " << node_info(insertBefore);
    LOG(INFO) << "create aten::tensor node: " << node_info(indices);
    return indices->output();
}

/**
* @brief 提取slice节点中的dim，start，end，step等信息，转化成slice tensor
* **/
Value* FuseCopy::convert_slice_to_index(Node* slice, Value* size, Node* insertBefore) {
    // Create index tensor based on aten::slice node.
    auto graph = slice->owningGraph();
    WithInsertPoint guard(insertBefore);
    TORCH_INTERNAL_ASSERT((slice->inputs()).size() == 5);
    auto start = slice->inputs()[2];
    auto end = slice->inputs()[3];
    auto step = slice->inputs()[4];
    //auto index = graph->insert(aten::arange, {size});
    auto index = graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
    LOG(INFO) << "convert_slice_to_index before node: " << node_info(insertBefore);
    LOG(INFO) << "create aten::arange node: " << node_info(index->node());
    auto sliced_index_n = graph->create(aten::slice, {
            index,
            graph->insertConstant(at::Scalar(0)), 
            start,
            end,
            step});
    LOG(INFO) << "create aten::slice node: " << node_info(sliced_index_n);
    sliced_index_n->copyMetadata(insertBefore);
    auto sliced_index = sliced_index_n->insertBefore(insertBefore)->output();
    return sliced_index;
}

//torch.version >= 1.12, Source api发生调整,兼容之
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 12
#define NODE_SOURCE_TEXT(name)  \
    name->text_str()
#else
#define NODE_SOURCE_TEXT(name)  \
    name->text()
#endif

/**
* @brief 找到跟 copy_ 或者 index_put_ 等op相关联的 slice op
*        他们来自python的同一行代码，
*        是为了合作完成list 或者 tensor 的切片功能
*        比如 y = x[1:3, 0] 这样的形式
// Example graph: 
//    %306 : Float(*, 16, 64, 16, 16) = aten::slice(%out.4, %0, %none, %none, %1) 
//    %307 : Float(*, 15, 64, 16, 16) = aten::slice(%306, %1, %none, %11, %1) 
//    %308 : Float(*, 15, 8, 16, 16) = aten::slice(%307, %2, %none, %y, %1) 
//    %309 : Tensor = aten::copy_(%308, %305, %false)  
* **/
std::vector<Node*> FuseCopy::fetch_slice_and_select_pattern(const Node* node) {
    TORCH_INTERNAL_ASSERT(node->kind() == aten::index_put ||
        node->kind() == aten::index_put_ ||
        node->kind() == aten::copy_);
    const auto& node_source = node->sourceRange().source();

    std::vector<Node*> slice_and_select_nodes;
    auto src_node = node->input(0)->node();
    while (src_node) {
        auto& src_node_source = src_node->sourceRange().source();         
        if ((src_node->kind() == aten::slice || src_node->kind() == aten::select) &&
            NODE_SOURCE_TEXT(node_source) == NODE_SOURCE_TEXT(src_node_source) &&
            node_source->starting_line_no() == src_node_source->starting_line_no()) {
            slice_and_select_nodes.emplace_back(src_node);
            //常常是连续的slice
            src_node = src_node->input(0)->node();
        } else {
            src_node = nullptr;
        }
    }
    return slice_and_select_nodes;
}

/**
* @brief 把相关联的slice 和 select 整合成 indices:
* **/ 
std::unordered_map<int64_t, ConvertedIndex> FuseCopy::merge_slice_and_select_to_indices(
                                        Graph* graph,
                                        Node* index_put_node,
                                        const std::vector<Node*>& slice_and_select_nodes,
                                        Value* orig_data) {

    std::unordered_map<int64_t, ConvertedIndex> dim_index_map;
    int64_t cur_dim = 0;
    /* dim_offset 的意义: 当select 和 slice 混合出现，完成对 tensor 的切片功能时，
        由于select 有降维的效果，aten::select 后面跟的 op (包括 select 和 slice) 的 dim 信息会被影响到，
        所以需要根据aten::select 已经出现的次数，对后续 op 的dim信息进行修正。
    */
    int64_t dim_offset = 0;
    const auto orig_tensor_indices = index_put_node->input(1)->node()->inputs();
    // slice_and_select_nodes 的添加过程是逆序的，所以逆向迭代vector 内的 slice 和 select 节点。
    for (auto it = slice_and_select_nodes.rbegin(); it != slice_and_select_nodes.rend(); ++it) {
        Node* node = *it;
        LOG(INFO) << "handle slice or select node info: " << node_info(node);
        //int64_t dim = node->inputs().at(1)->node()->t(attr::value).item().toLong();
        int64_t dim = toIValue(node->inputs().at(1)).value().toInt();
        if (dim < 0) {
            auto input_type = orig_data->type()->expect<TensorType>();
            if (input_type->dim().has_value()) {
                auto rank = static_cast<int64_t>(input_type->dim().value());
                dim = dim + rank - dim_offset;
            } else {
                std::cerr << "Error: Poros handle index Ops - Cannot export ellipsis indexing for input "
                        << "of unknown rank.";
            }
        }

        dim = dim + dim_offset;
        while (cur_dim < dim) {
            if (cur_dim - dim_offset >= (int64_t)orig_tensor_indices.size() ||
                index_put_node->input(1)->node()->input(cur_dim - dim_offset)->node()->mustBeNone()) {
                auto size = create_size_of_dim(orig_data, cur_dim, index_put_node);
                WithInsertPoint guard(index_put_node);
                //auto index_tensor = graph->insert(aten::arange, {size});
                auto index_tensor = graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
                LOG(INFO) << "create aten::arange node: " << node_info(index_tensor->node());
                dim_index_map.emplace(std::piecewise_construct, std::forward_as_tuple(cur_dim),
                                    std::forward_as_tuple(index_tensor, aten::slice));
            } else if (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
                dim_index_map.emplace(std::piecewise_construct, std::forward_as_tuple(cur_dim),
                                    std::forward_as_tuple(orig_tensor_indices[cur_dim - dim_offset], aten::index));
            }
            cur_dim++;
        }

        AT_ASSERT(cur_dim == dim);
        LOG(INFO) << "cur_dim info: " << cur_dim  <<  ", dim_offset: " << dim_offset;

        if (node->kind() == aten::slice) {
            auto size = create_size_of_dim(orig_data, dim, index_put_node);
            auto index_tensor = convert_slice_to_index(node, size, index_put_node);
            dim_index_map.emplace(std::piecewise_construct, std::forward_as_tuple(dim),
                                std::forward_as_tuple(index_tensor, aten::slice));
        } else if (node->kind() == aten::select) {
            auto index_tensor = convert_select_to_index(node->input(2), index_put_node);
            dim_index_map.emplace(std::piecewise_construct, std::forward_as_tuple(dim),
                                std::forward_as_tuple(index_tensor, aten::select));
            dim_offset++;
        } else {
            AT_ERROR("Unexpected node kind ", node->kind().toDisplayString(), " Expected aten::slice or aten::select.");
        }
        cur_dim++;
    }
    
    while (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
        dim_index_map.emplace(std::piecewise_construct, std::forward_as_tuple(cur_dim),
                            std::forward_as_tuple(orig_tensor_indices[cur_dim - dim_offset], aten::index));
        cur_dim++;
    }
    // Each dimension should have its associated index tensor.
    AT_ASSERT((int64_t)dim_index_map.size() == cur_dim);
    return dim_index_map;
}

std::vector<Value*> FuseCopy::reshape_to_advanced_indexing_format(Graph* graph, Node* index_put_node,
                    std::unordered_map<int64_t, ConvertedIndex>& dim_index_map) {
    std::vector<Value*> indices;
    size_t min_index_dim = dim_index_map.size();
    size_t max_index_dim = 0;
    size_t tensor_ind_count = 0;
    for (size_t i = 0; i < dim_index_map.size(); ++i) {
        auto index_i = dim_index_map.find(i);
        AT_ASSERT(index_i != dim_index_map.end());
        if (index_i->second.orig_node_kind == aten::index) {
            if (i < min_index_dim)
                min_index_dim = i;
            if (i > max_index_dim)
                max_index_dim = i;
            tensor_ind_count++;
        }
    }
    
    if (((max_index_dim - min_index_dim + 1) != tensor_ind_count) && tensor_ind_count != 0) {
        AT_ERROR("Only consecutive 1-d tensor indices are supported in exporting aten::index_put to POROS.");
    }
    
    size_t tensor_ind_offset = tensor_ind_count == 0 ? 0 : tensor_ind_count - 1;
    WithInsertPoint guard(index_put_node);
    for (size_t i = 0; i < dim_index_map.size(); ++i) {
        size_t ind_size = 0;
        auto index_i = dim_index_map.find(i);
        AT_ASSERT(index_i != dim_index_map.end());
        Value* index = index_i->second.index;
        switch (index_i->second.orig_node_kind) {
            case aten::select:
            case aten::slice: {
                if (i < min_index_dim) {
                    ind_size = dim_index_map.size() - tensor_ind_offset - i;
                } else {
                    ind_size = dim_index_map.size() - i;
                }
                break;
            }
            case aten::index: {
                ind_size = dim_index_map.size() - tensor_ind_offset - min_index_dim;
                break;
            }
            default:
                AT_ERROR("Unexpected node kind ", index_i->second.orig_node_kind);
        }
        
        if (ind_size != 1) {
            std::vector<int64_t> view_shape(ind_size, 1);
            view_shape[0] = -1;
            auto unsqueezed_index = graph->insert(aten::view, {index, view_shape});
            LOG(INFO) << "create aten::view node: " << node_info(unsqueezed_index->node());
            indices.emplace_back(unsqueezed_index);
        } else {
            indices.emplace_back(index);
        }
    }
    return indices;
}

/**
* @brief 针对aten::index_put / aten::index_put_ 的处理:
*        将跟他们相关联的slice 和 select 节点整合到一起, 提取indices信息，重新写个index_put。
* **/
bool FuseCopy::prepare_index_put(Node* index_put_node) {
    LOG(INFO) << "prepare for index put node: " << node_info(index_put_node);
    TORCH_INTERNAL_ASSERT(index_put_node->kind() == aten::index_put ||
                        index_put_node->kind() == aten::index_put_);
    //找到相关联的slice 和 select
    std::vector<Node*> slice_and_select_nodes = fetch_slice_and_select_pattern(index_put_node);
    if (slice_and_select_nodes.size() == 0) {
        return false;
    }
    LOG(INFO) << "slice_and_select_nodes_size: " << slice_and_select_nodes.size();
    Node* last_node = slice_and_select_nodes.size() > 0 ? slice_and_select_nodes.back() : index_put_node;
    //找到最原始的那个被切片的value， 具体到example graph中，原始value 为 %out.4。
    Value* orig_data = last_node->input(0);
    //当index_put 所在的node 与被改变的value不在一个block的时候，跳过这种情况。
    if (orig_data->node()->owningBlock() != index_put_node->owningBlock()) {
        LOG(INFO) << "orig data comes from different block, bypass this situation";
        return false;
    }

    auto graph = index_put_node->owningGraph();
    //对value进行处理。
    adjust_value(graph, index_put_node, slice_and_select_nodes, orig_data);

    //把slice和select操作转变成indices。
    std::unordered_map<int64_t, ConvertedIndex> dim_index_map = 
        merge_slice_and_select_to_indices(graph, index_put_node, slice_and_select_nodes, orig_data);
    

    std::vector<Value*> indices = reshape_to_advanced_indexing_format(graph, index_put_node, dim_index_map);

    // Create new index_put node with converted indices.
    const auto list_indices = graph->createList(OptionalType::ofTensor(), indices)
                                        ->insertBefore(index_put_node)->output();
    LOG(INFO) << "create tensorlist node: " << node_info(list_indices->node());
    auto new_index_put_node = graph->create(aten::index_put, 
                                {orig_data, list_indices, index_put_node->input(2), index_put_node->input(3)});
    LOG(INFO) << "create aten::index_put node: " << node_info(new_index_put_node);
    new_index_put_node->insertBefore(index_put_node);
    new_index_put_node->copyMetadata(index_put_node);
    auto new_index_put = new_index_put_node->output();
    new_index_put->copyMetadata(index_put_node->output());
    index_put_node->output()->replaceAllUsesWith(new_index_put);
    orig_data->replaceAllUsesAfterNodeWith(index_put_node, new_index_put);
    record_transform(index_put_node)->to(new_index_put_node);
    index_put_node->destroy();
    return true;
}


/**
* @brief 针对aten::copy_的处理: 将其用 index_put_ 代替。
* 此步骤中用到的dummylist 只是一个”站位符“，不能真正用于index_put
* prepare_index_put 会找到真正的 index信息。
* **/ 

// Example:
//    %out: Tensor = aten::copy_(%self, %src, %non_blocking)
//
// After this prepare function:
//    %dummylist : Tensor?[] = prim::ListConstruct()
//    %newout: Tensor = aten::index_put_(%self, %dummylist, %src, %non_blocking)
    bool FuseCopy::prepare_copy(Node* node) {
    TORCH_INTERNAL_ASSERT(node->kind() == aten::copy_);
    LOG(INFO) << "prepare for copy node: " << node_info(node);

    //找到相关联的slice 和 select
    std::vector<Node*> slice_and_select_nodes = fetch_slice_and_select_pattern(node);
    if (slice_and_select_nodes.size() == 0) {
        return false;
    }

    //找到最原始的那个被切片的value， 具体到example graph中，原始value 为 %out.4。先解决引用语义的问题。
    Node* last_node = slice_and_select_nodes.back();
    Value* orig_data = last_node->input(0);
    //当copy_ 所在的node 与被改变的value不在一个block的时候，跳过这种情况。
    if (orig_data->node()->owningBlock() != node->owningBlock()) {
        LOG(INFO) << "orig data comes from different block, bypass this situation";
        return false;
    }
    orig_data->replaceAllUsesAfterNodeWith(node, node->output());

    //做index_put 的替换
    WithInsertPoint guard(node);
    auto graph = node->owningGraph();
    Value* dummy_list = graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))->output();
    
    // 当value的size跟self的size不一致的时候，需要对齐两者的size信息，
    // 尝试在此处直接用expand_as, 发现单测无法通过，因为index_put支持value的rank为0的情况，
    // 此时需要修改index_put converter 的实现，兼容value 的rank为0的情况。
    // Value* expanded_value = graph->insert(aten::expand_as, 
    //                                     {node->input(1), orig_data});
    // expanded_value->node()->setSourceRange(node->sourceRange());
    // expanded_value->copyMetadata(node->input(1));
    // expanded_value->node()->copyMetadata(node);

    Value* index_put = graph->insert(aten::index_put_, 
                                    {node->input(0), dummy_list, node->input(1), node->input(2)});
    index_put->node()->copyMetadata(node);
    index_put->copyMetadata(node->output());
    node->output()->replaceAllUsesWith(index_put);

    record_transform(node)->to(index_put->node());
    bool changed = prepare_index_put(index_put->node());
    if (changed == true) {
        node->destroy();
    }
    return changed;
}

/**
 * search for aten::copy_  or aten::index_put patten for fuse
 * @param block
 * @return true if fuse success
 */
bool FuseCopy::try_to_fuse_copy(torch::jit::Block *block) {
    bool graph_changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        Node* node = *it;
        it++; // node n can be destroyed

        auto nkind = node->kind();
        //sub_block situation
        if (nkind == prim::If || nkind == prim::Loop) {
            for (Block* sub_block : node->blocks()) {
                try_to_fuse_copy(sub_block);
            }
        } else {
            if (nkind == aten::copy_) {
                LOG(INFO) << "copy situation meet";
                graph_changed |= prepare_copy(node);             
            } else if (nkind == aten::index_put || nkind == aten::index_put_) {
                LOG(INFO) << "index_put or index_put situation meet";
                graph_changed |= prepare_index_put(node);
            }
        }
    }
    return graph_changed;
}

REGISTER_OP_FUSER(FuseCopy)

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu