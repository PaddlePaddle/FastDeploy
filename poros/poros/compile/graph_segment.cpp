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
* @file graph_segment.cpp
* @author tianjinjin@baidu.com
* @author tianshaoqing@baidu.com
* @date Fri Mar 19 19:18:20 CST 2021
* @brief 
**/
#include "poros/compile/graph_segment.h"

//pytorch
#include <torch/script.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
// #include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include "poros/context/poros_global.h"
#include "poros/lowering/lowering_pass.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;

Value* broadcast_sizes(at::ArrayRef<Value*> sizes) {
    AT_ASSERT(!sizes.empty());
    Graph* graph = sizes[0]->owningGraph();
    Node* broadcast_n =
        graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
    broadcast_n->output()->setType(ListType::ofInts());
    return broadcast_n->output();
}

struct PorosGraphSegment {
    using FusionCallback = std::function<bool(Node*)>;
    Block* block_;
    std::unique_ptr<AliasDb> aliasDb_;
    std::shared_ptr<Graph> graph_;
    Symbol kind_ = prim::CudaFusionGroup;
    IEngine* engine_;
    
    PorosGraphSegment(Block* block, std::shared_ptr<Graph> graph, IEngine* engine)
        : block_(block), graph_(std::move(graph)), engine_(engine) {}

    //判断一个节点和它某个输入的关系，是否该输入的所有消费者已经在这group里了。
    bool all_users_are_this_cunsumer(Node* consumer, Value* producer) {
        Node* defining_node = producer->node();
        for (Value* o : defining_node->outputs()) {
            for (auto u : o->uses()) {
                if (u.user != consumer &&
                    !(u.user->matches("aten::size(Tensor self) -> int[]"))) {
                    return false;
                }
            }
        }
        return true;
    }

    //判断给定的一个节点（node)，当前engine是否支持。
    bool is_node_fusable(const Node* node) {
        //针对aten::append需要额外的判断条件。
        //当aten::append位于某个block，而它可能改变其parentblock中的ListConstruct的产出的时候，整个逻辑就gg了。
        //本质上，这还是inplace 语义导致的问题。
        //poros需要针对 inplace语义的算子，做更好的预处理逻辑。
        // if ((node->kind() == aten::append) && engine_->is_node_supported(node) &&
        //     node->inputs().at(0)->node()->kind() == prim::ListConstruct) {
        //     const Node* mutable_node = node->inputs().at(0)->node();
        //     for (auto &use: mutable_node->output()->uses()) {
        //         if (use.user->owningBlock() != mutable_node->owningBlock()) {
        //             LOG(WARNING) << "opps! meet mutable aten::append: " << node_info(node);
        //             return false;
        //         }
        //     }
        //     return true;
        // }
        // loop和if子block里的mutable op还没有串联，fuse会有问题，先禁用 04.22
        if (PorosGlobalContext::instance().supported_mutable_ops_set.count(node->kind()) > 0 
                                                        && engine_->is_node_supported(node)) {
            if (node->owningBlock() != node->owningGraph()->block()) {
                LOG(WARNING) << "Graph fuser meets mutable op in sub_block, which is not"
                " yet supported. Node info: " << node_info(node);
                return false;
            }
        }

        // aten::__getitem__ idx参数不支持非constant类型
        if (node->kind() == torch::jit::aten::__getitem__) {
            if (node->inputs().size() == 2 && 
                node->input(1)->node()->kind() != torch::jit::prim::Constant) {
                LOG(WARNING) << "The index input of aten::__getitem__ is not supported as non-constant type.";
                return false;
            }
        }

        // aten::_set_item idx参数不支持非constant类型
        if (node->kind() == torch::jit::aten::_set_item) {
            if (node->inputs().size() == 3 && 
                node->input(1)->node()->kind() != torch::jit::prim::Constant) {
                LOG(WARNING) << "The index input of aten::_set_item is not supported as non-constant type.";
                return false;
            }
        }

        if (node->kind() == kind_ || engine_->is_node_supported(node)) {
            return true;
        }
        return false;
    }

    //TODO: this should be better
    //判断给定的一个节点（node），是否可以fuse到已有的节点组（fusion）里面去。
    bool is_node_fusable(const Node* fusion, const Node* node) {
        //对prim::ListConstruct 这种有引用语义的，需要更严格的校验。
        // if (node->kind() == prim::ListConstruct && is_node_fusable(node)) {
        //     for (auto &use: node->output()->uses()) {
        //         if (use.user->owningBlock() != fusion->owningBlock()) {
        //             LOG(WARNING) << "opps! meet mutable ListConstruct: " << node_info(node);
        //             return false;
        //         }
        //     }
        //     return true;
        // }
        if (is_node_fusable(node)) {
            return true;
        }
        return false;
    }

    //返回给定节点的子图
    Graph& get_subgraph(Node* node) {
        AT_ASSERT(node->kind() == kind_);
        return *node->g(attr::Subgraph);
    }
    
    // 合并两个graph
    void merge_fusion_groups(Node* consumer_group, Node* producer_group) {
        // Now we have two fusion groups!
        // Revert the fusion - place all inner nodes of producer back in the outer
        // graph.
        std::vector<Node*> temporary_nodes;
        Graph* producer_subgraph = &get_subgraph(producer_group);

        // Initialize a map of inner graph values to outer graph values
        std::unordered_map<Value*, Value*> inner_to_outer;
        at::ArrayRef<Value*> inner_inputs = producer_subgraph->inputs();
        at::ArrayRef<Value*> outer_inputs = producer_group->inputs();
        for (size_t i = 0; i < inner_inputs.size(); ++i) {
            inner_to_outer[inner_inputs[i]] = outer_inputs[i];
        }

        // Clone all nodes
        for (auto inner : producer_subgraph->nodes()) {
            Node* outer = block_->owningGraph()->createClone(
                inner, [&](Value* k) -> Value* { return inner_to_outer.at(k); });
            for (size_t i = 0 ; i < outer->inputs().size(); i++){
                outer->input(i)->setType(inner->input(i)->type());
            }
            outer->insertBefore(producer_group);
            temporary_nodes.emplace_back(outer);
            at::ArrayRef<Value*> inner_outputs = inner->outputs();
            at::ArrayRef<Value*> outer_outputs = outer->outputs();
            for (size_t i = 0; i < inner_outputs.size(); ++i) {
                inner_to_outer[inner_outputs[i]] = outer_outputs[i];
            }
        }

        // Replace uses of producer_group outputs and destroy the producer
        at::ArrayRef<Value*> subgraph_outputs = producer_subgraph->outputs();
        for (size_t i = 0; i < subgraph_outputs.size(); ++i) {
            Value* outer_output = inner_to_outer.at(subgraph_outputs[i]);
            producer_group->outputs()[i]->replaceAllUsesWith(outer_output);
            update_global_context(producer_group->outputs()[i], outer_output);
        }

        // Inline the temporary nodes into the first group
        Graph* consumer_subgraph = &get_subgraph(consumer_group);
        for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend(); ++it) {
            Node* node = *it;
            Node* merged = merge_node_into_group(consumer_group, node);
            // If any of the outputs are still used then we need to add them
            at::ArrayRef<Value*> outputs = node->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                Value* output = outputs[i];
                if (output->uses().size() == 0) {
                    continue;
                }
                consumer_subgraph->registerOutput(merged->outputs()[i]);
                Value* new_output = consumer_group->addOutput();
                output->replaceAllUsesWith(new_output);
                update_global_context(output, new_output);
                new_output->setType(output->type());
            }
            node->destroy();
        }
        update_global_list_size_map_node_key_context(producer_group, consumer_group);
        producer_group->destroy();
        producer_group = nullptr; // Just to get a clear error in case someone uses it
    }

    Node* merge_node_into_group(Node* group, Node* to_merge_node) {
        AT_ASSERT(to_merge_node->kind() != kind_);
        Graph& subgraph = get_subgraph(group);

        // map from nodes in the surrounding graph to parameters in the fusion
        // group's subgraph that correspond to them
        std::unordered_map<Value*, Value*> inputs_map;
        size_t i = 0;
        size_t tensor_insert_idx = 0;
        AT_ASSERT(group->inputs().size() == subgraph.inputs().size());
        for (auto input : group->inputs()) {
            inputs_map[input] = subgraph.inputs()[i++];
            if (input->type()->isSubtypeOf(TensorType::get())) {
                tensor_insert_idx = i;   //真的要单独搞个tensor_index_idx 么？
            }
        }

        WithInsertPoint guard(*subgraph.nodes().begin());
        for (auto input : to_merge_node->inputs()) {
            if (inputs_map.count(input) == 0) {
                // TODO: we are following the convention for no good reason;
                //       we don't need tensor to come before any other inputs.
                if (input->type()->isSubtypeOf(TensorType::get())) {
                    Value* in_group = subgraph.insertInput(tensor_insert_idx);
                    in_group->setType(input->type());
                    inputs_map[input] = in_group;
                    group->insertInput(tensor_insert_idx, input);
                    tensor_insert_idx++;
                } else if (
                    // TODO: extend the supporting inputs here.
                    (input->type()->isSubtypeOf(FloatType::get()) &&
                    input->node()->kind() != prim::Constant) ||
                    (to_merge_node->kind() == aten::_grad_sum_to_size &&
                    input->type()->isSubtypeOf(ListType::ofInts()))) {
                    Value* in_group = subgraph.addInput();
                    in_group->setType(input->type());
                    inputs_map[input] = in_group;
                    group->addInput(input);
                } else if (input->node()->kind() == prim::Constant) {
                    // inline the constants directly in the body of the fused group.
                    Node* in_const =
                        subgraph.createClone(input->node(), [](Value*) -> Value* {
                            throw std::runtime_error("unexpected input");
                        });
                    subgraph.insertNode(in_const);
                    inputs_map[input] = in_const->output();
                } else {
                    Value* in_group = subgraph.addInput();
                    in_group->setType(input->type());
                    inputs_map[input] = in_group;
                    group->addInput(input);
                }
            }
        }

        // for (auto input : to_merge_node->inputs()) {
        //     update_inputs_map(to_merge_node, input);
        // }
        
        // copy n into the graph, remapping its inputs to internal nodes
        Node* in_graph = subgraph.createClone(
            to_merge_node, [&](Value* k) -> Value* { return inputs_map[k]; }, true);

        at::ArrayRef<Value*> inputs = group->inputs();
        for (size_t i = 0; i < to_merge_node->outputs().size(); ++i) {
            auto it = std::find(inputs.begin(), inputs.end(), to_merge_node->outputs()[i]);
            if (it != inputs.end()) {
                size_t p = it - inputs.begin();
                group->removeInput(p);
                subgraph.inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
                subgraph.eraseInput(p);
            }
        }
        return subgraph.insertNode(in_graph);
    }

    //将node转化为一个subgraph
    Node* create_singleton_fusion_group(Node* n) {
        Node* group = block_->owningGraph()->createWithSubgraph(kind_);
        // propogate position information for the new node so we can always
        // have a valid mapping
        group->insertBefore(n);
        Node* mergedNode = merge_node_into_group(group, n);
        if (mergedNode->outputs().size() == 1) {
            get_subgraph(group).registerOutput(mergedNode->output());
            Value* sel = group->addOutput();
            sel->copyMetadata(n->output());
            n->replaceAllUsesWith(group);
            update_global_context(n->output(), sel);
        //fix bug: handle situation when node has more than one output situation.
        } else {
            for (size_t index = 0; index <  mergedNode->outputs().size(); index++) {
                get_subgraph(group).registerOutput(mergedNode->outputs().at(index));
                Value* new_value = group->insertOutput(index)->copyMetadata(n->outputs().at(index));
                n->outputs().at(index)->replaceAllUsesWith(new_value);
                update_global_context(n->outputs().at(index), new_value);
            }
        }
        update_global_list_size_map_node_key_context(n, group);
        n->destroy();
        return group;
    }

    at::optional<Node*> try_fuse(Node* consumer, Value* producer) {

        LOG(INFO) << "[try_fuse] consumer: " << node_info(consumer);
        LOG(INFO) << "[try_fuse] producer: " << node_info(producer->node());

        bool shouldFuse =
            //TODO: check carefully later
            is_node_fusable(consumer, producer->node()) &&
            // Rearrange nodes such that all uses of producer are after the
            // consumer. Fusion will rewrite those later uses to use the version of
            // producer generated by the fused blob. In this case, producer becomes
            // an output of the fusion group.
            aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

        if (producer->node()->kind() == prim::Constant) {
            shouldFuse = true;
        }

        if (!shouldFuse) {
            LOG(INFO) << "[try_fuse Fail] should not fuse";
            return at::nullopt;
        }

        Node* group = consumer;
        if (producer->node()->kind() == kind_) {
            if (consumer->kind() != kind_) {
                group = create_singleton_fusion_group(consumer);
                // should not update here cause consumer has destroyed.
                // update_global_list_size_map_node_key_context(consumer, group);
            }
            merge_fusion_groups(group, producer->node());
            LOG(INFO) << "[try_fuse Success] FusionGroup is: " << node_info(group);
            return group;
        }

        // TODO: pay attention here. we should check multi output situation carefully.
        if (producer->node()->outputs().size() != 1 &&
            !all_users_are_this_cunsumer(consumer, producer)) {
            LOG(INFO) << "[try_fuse Fail] Should not fuse, producer output sizes: " << producer->node()->outputs().size()
                    << ", and is all_users_are_this_cunsumer: " << all_users_are_this_cunsumer(consumer, producer);
            return at::nullopt;
        }

        if (consumer->kind() != kind_) {
            group = create_singleton_fusion_group(consumer);
            // should not update here cause consumer has destroyed.
            // update_global_list_size_map_node_key_context(consumer, group);
        }

        Node* merged = merge_node_into_group(group, producer->node());
        //support for constant input. cause we copy the input. no need to replace this.
        //TODO: pay attention here.  constant handle should be careful.
        if (producer->uses().size() != 0 &&
            producer->node()->kind() != prim::Constant) {
            get_subgraph(group).registerOutput(merged->output());
            Value* new_producer = group->addOutput();
            new_producer->copyMetadata(producer);
            producer->replaceAllUsesWith(new_producer);
            update_global_context(producer, new_producer);
        }
        update_global_list_size_map_node_key_context(producer->node(), group);
        if (producer->node()->kind() != prim::Constant) {
            producer->node()->destroy();
        }
        LOG(INFO) << "[try_fuse Success] FusionGroup is: " << node_info(group);
        return group;
    }

    value_list sort_reverse_topological(ArrayRef<Value*> inputs) {
        value_list result;
        for (auto i : inputs) {
            if ((i->node()->owningBlock() == block_) ||
                (i->node()->kind() == prim::Constant)) {
                result.push_back(i);
            }
        }
        // Sort in reverse topological order
        std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
            return a->node()->isAfter(b->node());
        });
        return result;
    }

    // returns where to continue scanning, and whether any fusion was made
    // todo  换条件
    std::pair<graph_node_list::iterator, bool> scan_node(Node* consumer, const std::string list_construct) {
        if (is_node_fusable(consumer)) {
            value_list inputs = sort_reverse_topological(consumer->inputs());
            for (Value* producer : inputs) {
                if ((list_construct == "input" && (producer->node()->kind() != prim::ListConstruct || consumer->kind() != prim::CudaFusionGroup)) ||
                        (list_construct == "output" && (consumer->kind() != prim::ListUnpack || producer->node()->kind() != prim::CudaFusionGroup)) ||
                        (producer->node()->kind() == prim::ListUnpack) || (consumer->kind() == prim::ListConstruct)) {
                    continue;
                }

                at::optional<Node*> fusion_group = try_fuse(consumer, producer);
                if (fusion_group) {
                    // after fusion, consumer moves into a FusionGroup, so inputs is no
                    // longer valid so we rescan the new FusionGroup for more fusions...
                    return std::make_pair(fusion_group.value()->reverseIterator(), true);
                }
            }
        }
        return std::make_pair(++consumer->reverseIterator(), false);
    }
  
    void refresh_aliasdb() {
        aliasDb_ = torch::make_unique<AliasDb>(graph_);
    }

    void optimize_fused_graphs() {
        for (Node* node : block_->nodes()) {
            if (node->kind() != kind_) {
                continue;
            }
            auto subgraph = node->g(attr::Subgraph);
            EliminateDeadCode(subgraph);
            EliminateCommonSubexpression(subgraph);
            ConstantPooling(subgraph);
        }
    }

    void run(const std::string list_construct="") {
        bool any_changed = true;
        while (any_changed) {
            any_changed = false;
            refresh_aliasdb();
            for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
                bool changed = false;
                std::tie(it, changed) = scan_node(*it, list_construct);
                any_changed |= changed;
            }
        }
        refresh_aliasdb();
        optimize_fused_graphs();

        //TODO: should I add this???
        // for (Node* n : block_->nodes()) {
        //     removeOutputsUsedOnlyInSize(n);
        // }

        for (Node* node : block_->nodes()) {
            for (Block* sub_block : node->blocks()) {
                PorosGraphSegment(sub_block, graph_, engine_).run(list_construct);
            }
        }
    }
};  // struct PorosGraphSegment

void gen_value_dyanamic_shape_of_tensorlist(torch::jit::Value* tensor_value, 
                                        size_t idx,
                                        std::map<int32_t, std::vector<c10::TensorTypePtr>> type_map) {
    auto &_value_dynamic_shape_map = PorosGlobalContext::instance()._value_dynamic_shape_map;
    
    // 这里在的tensor_value地址可能和预热时的profile value地址一样，所以直接覆盖
    ValueDynamicShape dynamic_shape;
    _value_dynamic_shape_map[tensor_value] = dynamic_shape;
    _value_dynamic_shape_map[tensor_value].is_dynamic = false;
    _value_dynamic_shape_map[tensor_value].max_shapes = type_map[idx][0]->sizes().concrete_sizes().value();
    _value_dynamic_shape_map[tensor_value].min_shapes = type_map[idx][0]->sizes().concrete_sizes().value();
    _value_dynamic_shape_map[tensor_value].opt_shapes = type_map[idx][0]->sizes().concrete_sizes().value();

    // max
    for (size_t i = 0; i < type_map[idx].size(); i++){
        std::vector<int64_t> tmp_max_shape = _value_dynamic_shape_map[tensor_value].max_shapes;
        for(size_t j = 0; j < tmp_max_shape.size(); j++){
            _value_dynamic_shape_map[tensor_value].max_shapes[j] = std::max(tmp_max_shape[j], type_map[idx][i]->sizes()[j].value());
        }
    }
    
    // min
    for (size_t i = 0; i < type_map[idx].size(); i++){
        std::vector<int64_t> tmp_min_shape = _value_dynamic_shape_map[tensor_value].min_shapes;
        for(size_t j = 0; j < tmp_min_shape.size(); j++){
            _value_dynamic_shape_map[tensor_value].min_shapes[j] = std::min(tmp_min_shape[j], type_map[idx][i]->sizes()[j].value());
        }
    }

    ValueDynamicShape& shape = _value_dynamic_shape_map[tensor_value];
    for (size_t i = 0; i < shape.max_shapes.size(); ++i) {
        if (shape.max_shapes[i] == shape.min_shapes[i] && shape.max_shapes[i] == shape.opt_shapes[i]) {
                shape.sizes.push_back(shape.max_shapes[i]);     
        } else {
            shape.sizes.push_back(-1);
            shape.is_dynamic = true;
        }
    }
}

// 此处为子图预判断
// 作用：由于AdjustmentListTensorOutput、AdjustmentListTensorInput、AdjustmentScalarInput处理子图时会额外增加一些节点，
// 对于一些可预知的必然回退的子图（例如：unconst node不足、不支持的输出类型等），我们就不去处理这个子图，以减少不必要的节点数增加。
bool cudafusion_should_be_handle(torch::jit::Node* node) {
    // 如果传入是其他节点，直接返回false
    if (node->kind() != torch::jit::prim::CudaFusionGroup) {
        return false;
    }
    
    int non_constant_node_num = 0;
    std::shared_ptr<torch::jit::Graph> subgraph = node->g(torch::jit::attr::Subgraph);
    Block* subblock = subgraph->block();
    // 子图太小不处理
    int32_t unconst_threshold = PorosGlobalContext::instance().get_poros_options().unconst_ops_thres;
    for (auto it = subblock->nodes().begin(); it != subblock->nodes().end(); ++it) {
        if (it->kind() != torch::jit::prim::Constant) {
            non_constant_node_num++;
            if (non_constant_node_num > unconst_threshold) {
                break;
            }
        }
    }
    if (non_constant_node_num <= unconst_threshold) {
        LOG(WARNING) << "Subgraph: " << node_info_with_attr(node) << " size is too small, No tactics will be applied to it.";
        return false;
    }
    return true;
}

void AddPackAndUnpack(std::shared_ptr<Graph>& group_graph, torch::jit::Value* value, size_t idx, torch::jit::Node* node, bool input=true){
    /* 在CudaFusionGroup内（ouput）/外（input），添加一个prim::ListUnpack；List[Tensor] -> Tensor、Tensor ...
    在CudaFusionGroup外（ouput）/内（input），添加一个prim::ListConstruct；Tensor、Tensor ... -> List[Tensor]*/

    // 根据input or output 选择不同的map, 如果input中存在该list，则优先使用input;这样避免了append给output带来的引用问题。
    LIST_SIZE_MAP list_size_map = {}; //PorosGlobalContext::instance()._list_size_map._list_size_map_input;
    TENSOR_LIST_TYPE_MAP list_tensor_type_map = {}; //PorosGlobalContext::instance()._list_size_map._list_tensor_type_map_input;

    if (input || PorosGlobalContext::instance()._list_size_map._list_size_map_input.count(value) != 0) {
        list_size_map = PorosGlobalContext::instance()._list_size_map._list_size_map_input;
        list_tensor_type_map = PorosGlobalContext::instance()._list_size_map._list_tensor_type_map_input;
        if (!input) {
            Node* input_node = list_size_map[value].begin()->first;
            list_size_map[value][node] = list_size_map[value][input_node];
            list_tensor_type_map[value][node] = list_tensor_type_map[value][input_node];
        }
    } 
    else {
        list_size_map = PorosGlobalContext::instance()._list_size_map._list_size_map_output;
        list_tensor_type_map = PorosGlobalContext::instance()._list_size_map._list_tensor_type_map_output;
    }

    // 获取该tensorlist的长度
    int list_size = 0;
    if (list_size_map.count(value) > 0) {
        if (list_size_map[value].count(node) > 0) {
            if (list_size_map[value][node].size() != 1) {
                 LOG(INFO) << "list " + value->debugName() << " has " << std::to_string(list_size_map[value].size()) << " lengths";
                 return;
            }
            list_size = *list_size_map[value][node].begin();
        }
        else {
            LOG(INFO) << "node is not in list_size_map, value: %" << value->debugName() << ", node info:" << node_info(node);
            throw c10::Error("node must be in list_size_map", "");
        }
    }
    else {
        LOG(INFO) << "value is not in list_size_map, value: %" << value->debugName();
        throw c10::Error("value must be in list_size_map", "");
    }
    if (list_size == 0) {
        LOG(INFO) << "The length of the output list is 0: " << node_info(node);
        return;
    }

    // 新建一个unpack_node  和 pack_node
    Node* unpack_node = group_graph->create(prim::ListUnpack, value);
    Node* pack_node = group_graph->create(prim::ListConstruct, unpack_node->outputs());
    pack_node->output(0)->setType(value->type());
    std::vector<TypePtr> guard_types;

    // 更新下，给后面的前置判断使用
    list_size_map[value][unpack_node] = {list_size};

    if(input) {
        pack_node->insertBefore(node);
        unpack_node->insertBefore(pack_node);
    }
    else {
        unpack_node->insertAfter(node);
        pack_node->insertAfter(unpack_node);
    }

    // 更新相关输入输出
    bool is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;  
    std::map<int32_t, std::vector<c10::TensorTypePtr>> type_map = list_tensor_type_map[value][node];
    pack_node->replaceInput(0, unpack_node->output(0));    
    unpack_node->replaceInput(0, value);
    unpack_node->output(0)->setType(type_map[0][0]);
    pack_node->input(0)->setType(type_map[0][0]);
    guard_types.push_back(type_map[0][0]);

    if (!input) {
        value->replaceAllUsesWith(pack_node->output(0));
        update_global_context(value, pack_node->output(0));
        unpack_node->replaceInput(0, value);
    }
    if (is_dynamic_shape && input) {
        gen_value_dyanamic_shape_of_tensorlist(unpack_node->output(0), 0, type_map);
    }
    
    for (int j = 0; j < list_size - 1; j++){
        unpack_node->insertOutput(j + 1);
        pack_node->insertInput(j + 1, unpack_node->output(j + 1));
        unpack_node->output(j + 1)->setType(type_map[j + 1][0]);
        pack_node->input(j + 1)->setType(type_map[j + 1][0]);
        guard_types.push_back(type_map[j + 1][0]);
        if (is_dynamic_shape && input) {
            gen_value_dyanamic_shape_of_tensorlist(unpack_node->output(j + 1), j + 1, type_map);
        }
    }

    if (input) {
        node->replaceInput(idx, pack_node->output(0));
    }
    unpack_node->tys_(attr::types, guard_types);
}

void AdjustmentListTensorInput(std::shared_ptr<Graph>& group_graph, Block* block) {
    /*把tensor list类型的输入纠正为多个tensor的输入，使其适配tensorrt的输入类型*/
    graph_node_list nodes = block->nodes();
    for(auto it = nodes.begin(); it != nodes.end(); it++){
        for (Block* subblock : it->blocks()) {
            AdjustmentListTensorInput(group_graph, subblock);
        }
        if (it->kind() == prim::CudaFusionGroup) {
            if (!cudafusion_should_be_handle(*it)) {
                continue;
            }
            at::ArrayRef<Value*> inputs = it->inputs();
            for (size_t i = 0; i < inputs.size(); i++){
                if(inputs[i]->type()->str() == "Tensor[]") {
                    LOG(INFO) << "Adjustment Tensor[] input %" << inputs[i]->debugName();
                    AddPackAndUnpack(group_graph, inputs[i], i, *it, true);
                }
            }
        }
    }
}

void AdjustmentListTensorOutput(std::shared_ptr<Graph>& group_graph, Block* block) {
    /*把tensor list类型的输出纠正为多个tensor的输入，使其适配tensorrt的输出类型*/
    graph_node_list nodes = block->nodes();
    for(auto it = nodes.begin(); it != nodes.end(); it++){
        for (Block* subblock : it->blocks()) {
            AdjustmentListTensorOutput(group_graph, subblock);
        }
        if (it->kind() == prim::CudaFusionGroup) {
            if (!cudafusion_should_be_handle(*it)) {
                continue;
            }
            at::ArrayRef<Value*> outputs = it->outputs();
            for (size_t i = 0; i < outputs.size(); i++){
                if(outputs[i]->type()->str() == "Tensor[]") {
                    LOG(INFO) << "Adjustment Tensor[] output %" << outputs[i]->debugName();
                    AddPackAndUnpack(group_graph, outputs[i], i, *it, false);
                }
            }
        }
    }
}
// When cudafusiongroup subgraph input is int (or int[]) like:
// %1 : int = size(%x, %b)
// %4 : Tensor = prim::CudaFusionGroup(%1)
// or
// %1 : int[] = size(%x)
// %4 : Tensor = prim::CudaFusionGroup(%1)
//
// Then we insert aten::tensor and aten::IntImplicit (or prim::tolist) before the subgraph like:
// %1 : int = size(%x, %b)
// %2 : Tensor = aten::tensor(%1, %type, %device, %requires_grad)
// %3 : int = aten::IntImplicit(%2)
// %4 : Tensor = prim::CudaFusionGroup(%3)
// or
// %1 : int[] = size(%x)
// %2 : Tensor = aten::tensor(%1, %type, %device, %requires_grad)
// %3 : int = prim::tolist(%2, %dim, %type)
// %4 : Tensor = prim::CudaFusionGroup(%3)
// 
// Finally, merge the aten::IntImplicit (or prim::tolist) into the cudafusiongroup subgraph. The int input has been replaced by tensor.
bool AddInputTensorandScalarimplict(std::shared_ptr<Graph>& group_graph, torch::jit::Value* value, size_t idx, torch::jit::Node* node, IEngine* engine) {
    bool value_type_is_list = (value->type()->kind() == c10::TypeKind::ListType);
    int32_t list_size = 1;
    LIST_SIZE_MAP list_size_map = {};
    if (value_type_is_list) {
        // get list size
        list_size_map = PorosGlobalContext::instance()._list_size_map._list_size_map_input;
        if (list_size_map.count(value) > 0) {
            if (list_size_map[value].count(node) > 0) {
                if (list_size_map[value][node].size() != 1) {
                    LOG(WARNING) << "list " + value->debugName() << " has " << std::to_string(list_size_map[value].size()) << " lengths";
                    return false;
                }
                list_size = *list_size_map[value][node].begin();
            } else {
                LOG(WARNING) << "node is not in list_size_map, value: %" << value->debugName() << ", node info:" << node_info(node);
                return false;
            }
        } else {
            LOG(WARNING) << "value is not in list_size_map, value: %" << value->debugName();
            return false;
        }
    }
    // 检查全局_int_intlist_values_map中有无当前scalar值
    std::map<torch::jit::Value*, ValueDynamicShape>& int_intlist_values_map = PorosGlobalContext::instance()._int_intlist_values_map;
    if (value->type()->isSubtypeOf(c10::ListType::ofInts()) || value->type()->kind() == c10::TypeKind::IntType) {
        if (int_intlist_values_map.count(value) == 0) {
            LOG(WARNING) << "can't find max min opt of int(or int[]) %" << value->debugName();
            return false;
        }
    }

    std::map<torch::jit::Value*, ValueDynamicShape>& value_dynamic_shape_map = PorosGlobalContext::instance()._value_dynamic_shape_map;
    auto fuser = PorosGraphSegment(group_graph->block(), group_graph, engine);
    // 创建aten::tensor
    torch::jit::Node* tensor_node = group_graph->create(torch::jit::aten::tensor);
    tensor_node->insertBefore(node);
    tensor_node->addInput(value);
    // note: 没有setInsertPoint insertconstant默认到图的末尾插入节点
    // 但此处最好不要用setInsertPoint，当图发生变化导致point的点变化时候会出core
    // 建议使用”insertConstant之后moveBerfore“来代替”setInsertPoint后insertConstant“的操作
    // group_graph->setInsertPoint(tensor_node);
    // 创建aten::tensor dtype、device和requires_grad constant输入
    torch::jit::Value* type_value = nullptr;
    c10::optional<at::ScalarType> output_scalar_type;
    if (value_type_is_list) {
        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            type_value = group_graph->insertConstant(c10::ScalarType::Long);
            output_scalar_type = at::kLong;
        } else {
            type_value = group_graph->insertConstant(c10::ScalarType::Float);
            output_scalar_type = at::kFloat;
        }
    } else {
        if (value->type()->kind() == c10::TypeKind::IntType) {
            type_value = group_graph->insertConstant(c10::ScalarType::Int);
            output_scalar_type = at::kInt;
        } else {
            type_value = group_graph->insertConstant(c10::ScalarType::Float);
            output_scalar_type = at::kFloat;
        }
    }
    torch::jit::Value* device_value = nullptr;
    c10::optional<at::Device> output_device;
    if (PorosGlobalContext::instance().get_poros_options().device == Device::GPU) {
        device_value = group_graph->insertConstant(torch::Device(torch::DeviceType::CUDA, 0));
        output_device = torch::Device(at::kCUDA, 0);
    } else {
        torch::jit::IValue none_ivalue;
        device_value = group_graph->insertConstant(none_ivalue);
        output_device = torch::Device(at::kCPU);
    }
    torch::jit::Value* false_value = group_graph->insertConstant(false);
    // 没有setinsertpoint，insertconstant默认到了图的末尾，需要将constant移到tensor_node之前
    type_value->node()->moveBefore(tensor_node);
    device_value->node()->moveBefore(tensor_node);
    false_value->node()->moveBefore(tensor_node);

    tensor_node->addInput(type_value);
    tensor_node->addInput(device_value);
    tensor_node->addInput(false_value);
    // must set output type
    TypePtr output_type = c10::TensorType::create(output_scalar_type, 
                                output_device,
                                c10::SymbolicShape(std::vector<c10::optional<int64_t>>({list_size})),
                                std::vector<c10::Stride>({c10::Stride{0, true, 1}}),
                                false);
    tensor_node->output(0)->setType(output_type);
    // 更新value_dynamic_shape_map中aten::tensor output的max min opt值为int_intlist_values_map中的value对应的值。
    // 因为tensor_node->output(0)即将变为子图输入
    value_dynamic_shape_map[tensor_node->output(0)] = int_intlist_values_map[value];
    
    // 创建scalar implicit node
    // 如果是scalarlist
    if (value_type_is_list) {
        // 更新list_size_map中value在aten::tensor的list_size信息
        list_size_map[value][tensor_node] = {(int32_t)list_size};
        // int list create prim::tolist node
        torch::jit::Node* tolist_node = group_graph->create(torch::jit::prim::tolist);
        tolist_node->insertBefore(node);
        torch::jit::Value* dim_val = group_graph->insertConstant(int(1));
        torch::jit::Value* type_val = nullptr;
        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            // int list
            type_val = group_graph->insertConstant(int(0));
        } else {
            // float list
            type_val = group_graph->insertConstant(int(1));
        }
        tolist_node->addInput(tensor_node->output(0));

        dim_val->node()->moveBefore(tolist_node);
        type_val->node()->moveBefore(tolist_node);

        tolist_node->addInput(dim_val);
        tolist_node->addInput(type_val);

        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            tolist_node->output(0)->setType(c10::ListType::ofInts());
        } else {
            tolist_node->output(0)->setType(c10::ListType::ofFloats());
        }
        node->replaceInput(idx, tolist_node->output(0)); 

        // 手动更新map
        list_size_map[tolist_node->output(0)][tolist_node] = {(int32_t)list_size};
        list_size_map[tolist_node->output(0)][node] = {(int32_t)list_size};
        int_intlist_values_map[tolist_node->output(0)] = int_intlist_values_map[value];

        // 把tolist merge进子图中
        fuser.refresh_aliasdb();
        fuser.merge_node_into_group(node, type_val->node());
        fuser.merge_node_into_group(node, dim_val->node());
        fuser.merge_node_into_group(node, tolist_node);
        fuser.refresh_aliasdb();
        fuser.optimize_fused_graphs();

    // 如果输入是scalar
    } else {
        // int创建intimplicit
        torch::jit::Node* scalar_implicit_node = nullptr;
        if (value->type()->kind() == c10::TypeKind::IntType) {
            torch::jit::Node* intimplicit_node = group_graph->create(torch::jit::aten::IntImplicit, tensor_node->output(0));
            intimplicit_node->output(0)->setType(c10::IntType::get());
            intimplicit_node->insertBefore(node);
            node->replaceInput(idx, intimplicit_node->output(0));
            scalar_implicit_node = intimplicit_node;
        } else {
            // float创建FloatImplicit
            torch::jit::Node* floatimplicit_node = group_graph->create(torch::jit::aten::FloatImplicit, tensor_node->output(0));
            floatimplicit_node->output(0)->setType(c10::FloatType::get());
            floatimplicit_node->insertBefore(node);
            node->replaceInput(idx, floatimplicit_node->output(0));
            scalar_implicit_node = floatimplicit_node;
        }
        // 更新int_intlist_values_map
        int_intlist_values_map[scalar_implicit_node->output(0)] = int_intlist_values_map[value];
        fuser.refresh_aliasdb();
        fuser.try_fuse(node, node->input(idx));
        fuser.refresh_aliasdb();
        fuser.optimize_fused_graphs();
    }
    return true;
}

// 当子图输出是scalar（或scalar list）类型时，
// 创建aten::tensor与aten::IntImplicit（或prim::tolist）
// 然后将aten::tensor融合到子图中去
bool AddOutputTensorandScalarimplict(std::shared_ptr<Graph>& group_graph, torch::jit::Value* value, size_t idx, torch::jit::Node*& node, IEngine* engine) {
    bool value_type_is_list = (value->type()->kind() == c10::TypeKind::ListType);
    size_t list_size = 1;
    LIST_SIZE_MAP list_size_map = {};
    if (value_type_is_list) {
        // get list size
        if (PorosGlobalContext::instance()._list_size_map._list_size_map_input.count(value) != 0) {
            list_size_map = PorosGlobalContext::instance()._list_size_map._list_size_map_input;
            Node* input_node = list_size_map[value].begin()->first;
            list_size_map[value][node] = list_size_map[value][input_node];
        }
        else {
            list_size_map = PorosGlobalContext::instance()._list_size_map._list_size_map_output;
        }
        if (list_size_map.count(value) > 0) {
            if (list_size_map[value].count(node) > 0) {
                if (list_size_map[value][node].size() != 1) {
                    LOG(WARNING) << "list " + value->debugName() << " has " << std::to_string(list_size_map[value].size()) << " lengths";
                    return false;
                }
                list_size = *list_size_map[value][node].begin();
            } else {
                LOG(WARNING) << "node is not in list_size_map, value: %" << value->debugName() << ", node info:" << node_info(node);
                return false;
            }
        } else {
            LOG(WARNING) << "value is not in list_size_map, value: %" << value->debugName();
            return false;
        }
    }
    // 检查全局_int_intlist_values_map中有无当前scalar值
    std::map<torch::jit::Value*, ValueDynamicShape>& int_intlist_values_map = PorosGlobalContext::instance()._int_intlist_values_map;
    if (value->type()->isSubtypeOf(c10::ListType::ofInts()) || value->type()->kind() == c10::TypeKind::IntType) {
        if (int_intlist_values_map.count(value) == 0) {
            LOG(WARNING) << "can't find max min opt of int(or int[]) %" << value->debugName();
            return false;
        }
    }

    std::map<torch::jit::Value*, ValueDynamicShape>& value_dynamic_shape_map = PorosGlobalContext::instance()._value_dynamic_shape_map;
    auto fuser = PorosGraphSegment(group_graph->block(), group_graph, engine);
    // 创建aten::tensor
    torch::jit::Node* tensor_node = group_graph->create(torch::jit::aten::tensor);
    tensor_node->insertAfter(node);
    tensor_node->addInput(value);
    // 创建aten::tensor dtype、device和requires_grad constant输入
    torch::jit::Value* type_value = nullptr;
    c10::optional<at::ScalarType> output_scalar_type;
    if (value_type_is_list) {
        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            type_value = group_graph->insertConstant(c10::ScalarType::Long);
            output_scalar_type = at::kLong;
        } else {
            type_value = group_graph->insertConstant(c10::ScalarType::Float);
            output_scalar_type = at::kFloat;
        }
    } else {
        if (value->type()->kind() == c10::TypeKind::IntType) {
            type_value = group_graph->insertConstant(c10::ScalarType::Int);
            output_scalar_type = at::kInt;
        } else {
            type_value = group_graph->insertConstant(c10::ScalarType::Float);
            output_scalar_type = at::kFloat;
        }
    }
    torch::jit::Value* device_value = nullptr;
    c10::optional<at::Device> output_device;
    if (PorosGlobalContext::instance().get_poros_options().device == Device::GPU) {
        device_value = group_graph->insertConstant(torch::Device(torch::DeviceType::CUDA, 0));
        output_device = torch::Device(at::kCUDA, 0);
    } else {
        torch::jit::IValue none_ivalue;
        device_value = group_graph->insertConstant(none_ivalue);
        output_device = torch::Device(at::kCPU);
    }
    torch::jit::Value* false_value = group_graph->insertConstant(false);

    type_value->node()->moveBefore(tensor_node);
    device_value->node()->moveBefore(tensor_node);
    false_value->node()->moveBefore(tensor_node);
    
    tensor_node->addInput(type_value);
    tensor_node->addInput(device_value);
    tensor_node->addInput(false_value);

    // must set output type
    TypePtr output_type = c10::TensorType::create(output_scalar_type, 
                            output_device,
                            c10::SymbolicShape(std::vector<c10::optional<int64_t>>({list_size})),
                            std::vector<c10::Stride>({c10::Stride{0, true, 1}}),
                            false);
    tensor_node->output(0)->setType(output_type);

    value_dynamic_shape_map[tensor_node->output(0)] = int_intlist_values_map[value];

    // 创建scalar implicit node
    // 如果输入是scalarlist
    torch::jit::Node* tolist_node = nullptr;
    torch::jit::Node* scalar_implicit_node = nullptr;
    if (value_type_is_list) {
        // 更新list_size_map中value在aten::tensor子图的list_size信息
        list_size_map[value][tensor_node] = {(int32_t)list_size};
        // int list create prim::tolist node
        tolist_node = group_graph->create(torch::jit::prim::tolist);
        tolist_node->insertAfter(tensor_node);
        tolist_node->addInput(tensor_node->output(0));
        torch::jit::Value* dim_val = group_graph->insertConstant(int(1));
        torch::jit::Value* type_val = nullptr;
        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            // int list
            type_val = group_graph->insertConstant(int(0));
        } else {
            // float list
            type_val = group_graph->insertConstant(int(1));
        }

        dim_val->node()->moveBefore(tolist_node);
        type_val->node()->moveBefore(tolist_node);

        tolist_node->addInput(dim_val);
        tolist_node->addInput(type_val);

        if (value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            tolist_node->output(0)->setType(c10::ListType::ofInts());
        } else {
            tolist_node->output(0)->setType(c10::ListType::ofFloats());
        }
        value->replaceAllUsesAfterNodeWith(tolist_node, tolist_node->output(0));

        list_size_map[tolist_node->output(0)][tolist_node] = {(int32_t)list_size};
        // list_size_map中value有node概念，需要一个一个更新
        torch::jit::use_list tolist_node_user = tolist_node->output(0)->uses();
        for (size_t u = 0; u < tolist_node_user.size(); u++) {
            list_size_map[tolist_node->output(0)][tolist_node_user[u].user] = {(int32_t)list_size};
        }
        int_intlist_values_map[tolist_node->output(0)] = int_intlist_values_map[value];
    } else {
        // int create intimplicit node
        if (value->type()->kind() == c10::TypeKind::IntType) {
            torch::jit::Node* intimplicit_node = group_graph->create(torch::jit::aten::IntImplicit, tensor_node->output(0));
            intimplicit_node->output(0)->setType(c10::IntType::get());
            intimplicit_node->insertAfter(tensor_node);
            scalar_implicit_node = intimplicit_node;
        } else {
            // float create FloatImplicit node
            torch::jit::Node* floatimplicit_node = group_graph->create(torch::jit::aten::FloatImplicit, tensor_node->output(0));
            floatimplicit_node->output(0)->setType(c10::FloatType::get());
            floatimplicit_node->insertAfter(tensor_node);
            scalar_implicit_node = floatimplicit_node;
        }
        value->replaceAllUsesAfterNodeWith(scalar_implicit_node, scalar_implicit_node->output(0));
        // 更新int_intlist_values_map
        int_intlist_values_map[scalar_implicit_node->output(0)] = int_intlist_values_map[value];
    }
    // 为aten::tensor 创造子图，更新全局map，最后与node fuser
    fuser.refresh_aliasdb();
    torch::jit::Node* subgraph_node = fuser.create_singleton_fusion_group(tensor_node);
    fuser.merge_node_into_group(subgraph_node, type_value->node());
    fuser.merge_node_into_group(subgraph_node, device_value->node());
    fuser.merge_node_into_group(subgraph_node, false_value->node());
    // list_size_map只要更换节点就需要更新
    if (value_type_is_list) {
        list_size_map[value][subgraph_node] = {(int32_t)list_size};
    }
    value_dynamic_shape_map[subgraph_node->output(0)] = int_intlist_values_map[value];
    fuser.try_fuse(subgraph_node, subgraph_node->input(0));
    // 由于用了aten::tensor构造的子图来fuse，之前node的子图已经消失，需更新node为新融合的子图
    node = subgraph_node;
    fuser.refresh_aliasdb();
    fuser.optimize_fused_graphs();
    return true;
}

// 将子图的scalar输入转成tensor输入
bool adjust_scalar_input(std::shared_ptr<Graph>& group_graph, Block* block, IEngine* engine) {
    bool changed = false;
    graph_node_list nodes = block->nodes();
    for(auto it = nodes.begin(); it != nodes.end(); ) {
        Node* current_node = *it;
        it++;
        for (Block* subblock : current_node->blocks()) {
            changed |= adjust_scalar_input(group_graph, subblock, engine);
        }
        if (current_node->kind() == prim::CudaFusionGroup) {
            if (!cudafusion_should_be_handle(current_node)) {
                continue;
            }
            at::ArrayRef<Value*> subgraph_node_inputs = current_node->inputs();
            for (size_t i = 0; i < subgraph_node_inputs.size(); i++) {
                // todo: support float and float[]
                // mark by tsq 0713: loop中的scalar input可能会有问题，因为其记录的max min opt不一定真实，但目前没有遇到此类问题。
                if(subgraph_node_inputs[i]->type()->str() == "int" || subgraph_node_inputs[i]->type()->str() == "int[]") {
                    LOG(INFO) << "Adjustment subgraph: " << node_info_with_attr(current_node) << " scalar input %" << subgraph_node_inputs[i]->debugName();
                    std::string origin_input_debugname = subgraph_node_inputs[i]->debugName();
                    if (AddInputTensorandScalarimplict(group_graph, subgraph_node_inputs[i], i, current_node, engine)) {
                        LOG(INFO) << "Adjustment scalar input %" << origin_input_debugname << " to tensor %" 
                        << subgraph_node_inputs[i]->debugName() << " succeed!";
                        changed = true;
                    } else {
                        LOG(WARNING) << "Adjustment scalar input %" << origin_input_debugname << " failed!";
                    }
                }
            }
        }
    }
    return changed;
}

void AdjustmentScalarInput(std::shared_ptr<Graph>& group_graph, Block* block, IEngine* engine) {
    bool changed = false;
    changed = adjust_scalar_input(group_graph, block, engine);
    if (changed) {
        EliminateDeadCode(group_graph);
        EliminateCommonSubexpression(group_graph);
        ConstantPooling(group_graph);
    }
}

// 将子图的scalar输出转为tensor输出
bool adjust_scalar_output(std::shared_ptr<Graph>& group_graph, Block* block, IEngine* engine) {
    /*把tensor list类型的输入纠正为多个tensor的输入，使其适配tensorrt的输入类型*/
    bool changed = false;
    graph_node_list nodes = block->nodes();
    for(auto it = nodes.begin(); it != nodes.end(); ) {
        Node* current_node = *it;
        it++;
        for (Block* subblock : current_node->blocks()) {
            changed |= adjust_scalar_output(group_graph, subblock, engine);
        }
        if (current_node->kind() == prim::CudaFusionGroup) {
            if (!cudafusion_should_be_handle(current_node)) {
                continue;
            }

            for (size_t i = 0; i < current_node->outputs().size(); i++) {
                if (current_node->output(i)->type()->str() == "int" || current_node->output(i)->type()->str() == "int[]") {
                    // todo: support float and float[]
                    LOG(INFO) << "Adjustment subgraph: " << node_info_with_attr(current_node) << " scalar output %" << current_node->output(i)->debugName();
                    std::string origin_output_debugname =  current_node->output(i)->debugName();
                    if (AddOutputTensorandScalarimplict(group_graph, current_node->output(i), i, current_node, engine)) {
                        LOG(INFO) << "Adjustment scalar output %" << origin_output_debugname << " to tensor %" 
                        << current_node->output(i)->debugName() << " succeed!";
                        changed = true;
                        // 更新scalar output后子图会更新，在新的子图上继续寻找scalar output，直到所有输出都不是scalar。
                        i = 0;
                    } else {
                        LOG(WARNING) << "Adjustment scalar output %" << origin_output_debugname << " failed!";
                    }
                }
            }
        }
    }
    return changed;
}

void AdjustmentScalarOutput(std::shared_ptr<Graph>& group_graph, Block* block, IEngine* engine) {
    bool changed = false;
    changed = adjust_scalar_output(group_graph, block, engine);
    if (changed) {
        EliminateDeadCode(group_graph);
        EliminateCommonSubexpression(group_graph);
        ConstantPooling(group_graph);
    }
}

void peephole_optimize_shape_expressions(Block* block) {
    graph_node_list nodes = block->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        Node* node = *it;
        for (Block* subblock : node->blocks()) {
            peephole_optimize_shape_expressions(subblock);
        }
        if (node->kind() == prim::BroadcastSizes) {
            // Remove no-op broadcasts.
            if (node->inputs().size() == 1) {
                node->output()->replaceAllUsesWith(node->input());
                it.destroyCurrent();
                continue;
            }
            // Deduplicate inputs, but use their unique() values to ensure
            // this process only depends on the graph.
            std::map<size_t, Value*> unique_to_value;
            for (Value* input : node->inputs()) {
                unique_to_value.emplace(input->unique(), input);
            }
            if (unique_to_value.size() != node->inputs().size()) {
                std::vector<Value*> inputs;
                inputs.reserve(unique_to_value.size());
                for (auto& entry : unique_to_value) {
                    inputs.push_back(entry.second);
                }
                if (inputs.size() == 1) {
                    node->output()->replaceAllUsesWith(inputs[0]);
                } else {
                    WithInsertPoint insert_guard{node};
                    node->output()->replaceAllUsesWith(broadcast_sizes(inputs));
                }
                it.destroyCurrent();
                --it; // Revisit the node with deduplicated inputs
                continue;
            }
            // Remove compose simple chains of broadcasts into a single node.
            const auto& uses = node->output()->uses();
            if (uses.size() == 1 && uses[0].user->kind() == prim::BroadcastSizes) {
                Node* user = uses[0].user;
                user->removeInput(uses[0].offset);
                // NB: we don't care about deduplication in here, as we will visit user
                // later.
                for (Value* i : node->inputs()) {
                    user->addInput(i);
                }
                it.destroyCurrent();
            }
        }
    }
}   // peephole_optimize_shape_expressions

void guard_fusion_group(Node* fusion) {
    // Fixup types of the subgraph inputs
    std::vector<TypePtr> guard_types;
    std::vector<Value*> inputs_to_check;
    for (Value* input : fusion->inputs()) {
        // We only check inputs of the fusion group and expect NNC to infer
        // intermediates and outputs shapes
        if (!input->type()->cast<TensorType>()) {
            continue;
        }

        // note: modified from original implementation, we are guarding fusion
        //       outputs
        if (input->node()->kind() == prim::Constant) {
            continue;
        }
        inputs_to_check.push_back(input);
        guard_types.push_back(input->type());
    }
    if (!inputs_to_check.size()) {
        return;
    }

    Node* typecheck_node = fusion->owningGraph()
                                //this is not right, i should register my own type to torchscrilpt
                                //->create(prim::CudaFusionGuard, inputs_to_check, 1)
                                ->create(prim::FusionGroup, inputs_to_check, 1)
                                ->insertBefore(fusion);
    // fix output to BoolType
    typecheck_node->output()->setType(BoolType::get());
    Value* typecheck_result = typecheck_node->output();
    typecheck_node->tys_(attr::types, guard_types);

    std::unordered_map<Value*, Value*> typechecked_inputs;

    // Insert if block
    Node* versioning_if =
        fusion->owningGraph()
            ->create(prim::If, {typecheck_result}, fusion->outputs().size())
            ->insertAfter(typecheck_node);
    for (size_t idx = 0; idx < fusion->outputs().size(); ++idx) {
        versioning_if->output(idx)->setType(fusion->output(idx)->type());
        fusion->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
    }
    Block* true_block = versioning_if->addBlock();
    Block* false_block = versioning_if->addBlock();

    // Fill in the false block. It should contain the unoptimized
    // copy of the fused subgraph.
    auto& subgraph = *fusion->g(attr::Subgraph);
    WithInsertPoint guard(false_block->return_node());
    const std::vector<Value*> subgraph_outputs =
        insertGraph(*fusion->owningGraph(), subgraph, fusion->inputs());
    for (Value* output : subgraph_outputs) {
        false_block->registerOutput(output);
    }

    // types get copied to the fallback graph, so remove specializations before
    // replacing
    // TODO: this is not exposed here, I need to remove that before inserting the
    //       graph
    // removeTensorTypeSpecializations(false_block);
    replaceBlockWithFallbackGraph(false_block, fusion->inputs());

    // Fill in the true block. It has all inputs type-checked and its
    // body should be the fusion group node.
    fusion->moveBefore(true_block->return_node());
    for (Value* output : fusion->outputs()) {
        true_block->registerOutput(output);
    }
}  // guard_fusion_group

void guard_fusion_groups(Block* block) {
    std::vector<Node*> fusions;
    for (Node* n : block->nodes()) {
        for (Block* b : n->blocks()) {
            guard_fusion_groups(b);
        }
        if (n->kind() == prim::CudaFusionGroup) {
            fusions.push_back(n);
        }
    }
    for (Node* fusion : fusions) {
        guard_fusion_group(fusion);
    }
}   // guard_fusion_groups

}   // anonymous namespace

void graph_segment(std::shared_ptr<Graph>& graph, IEngine* engine) {

    GRAPH_DUMP("before PorosGraphSegment Graph: ", graph);
    PorosGraphSegment(graph->block(), graph, engine).run();
    GRAPH_DUMP("after PorosGraphSegment Graph: ", graph);
    //guard_fusion_groups(graph->block());

    //necessary passes after segmentation
    {
        torch::jit::EliminateCommonSubexpression(graph);
        torch::jit::EliminateDeadCode(graph);
        peephole_optimize_shape_expressions(graph->block());
        torch::jit::RemoveTensorTypeSpecializations(graph);
        GRAPH_DUMP("after necessary pass Graph: ", graph);
    }

    //necessary adjustmentations after segmentation
    {
        AdjustmentListTensorInput(graph, graph->block());
        PorosGraphSegment(graph->block(), graph, engine).run("input");
        GRAPH_DUMP("after AdjustmentListTensorInput Graph: ", graph);
        AdjustmentListTensorOutput(graph, graph->block());
        PorosGraphSegment(graph->block(), graph, engine).run("output");
        GRAPH_DUMP("after AdjustmentListTensorOutput Graph: ", graph);
        AdjustmentScalarInput(graph, graph->block(), engine);
        GRAPH_DUMP("after AdjustmentScalarInput Graph: ", graph);
        AdjustmentScalarOutput(graph, graph->block(), engine);
        GRAPH_DUMP("after AdjustmentScalarOutput Graph: ", graph);
    }
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
