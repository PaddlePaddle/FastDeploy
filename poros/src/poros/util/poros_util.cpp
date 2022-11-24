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
* @file poros_util.cpp
* @author tianjinjin@baidu.com
* @date Wed Apr  7 17:52:36 CST 2021
* @brief 
**/

#include "poros/util/poros_util.h"
#include "poros/log/poros_logging.h"
#include "poros/util/macros.h"
#include "poros/context/poros_global.h"

namespace baidu {
namespace mirana {
namespace poros {

int merge_graph_to_module(std::shared_ptr<torch::jit::Graph>& to_merge_graph, 
    torch::jit::Module& module,
    bool init_module_ptr) {
    if (init_module_ptr) {
        //先把model本身作为graph的第一个参数传进去, 此处勿忘!!!!!!
        auto self = to_merge_graph->insertInput(0, "self");
        self->setType(module._ivalue()->type());
    }
    
    auto new_method = module._ivalue()->compilation_unit()->create_function("forward", to_merge_graph);
    std::vector<c10::Argument> args;
    int index = 0;
    for (auto in : to_merge_graph->inputs()) {
        args.push_back(c10::Argument("input" + std::to_string(index), in->type()));
        index++;
    }

    index = 0;
    std::vector<c10::Argument> res;
    for (auto out : to_merge_graph->outputs()) {
        res.push_back(c10::Argument("output" + std::to_string(index), out->type()));
        index++;
    }
    auto schema = c10::FunctionSchema(new_method->name(), new_method->name(), args, res);
    module.type()->addMethod(new_method);
    new_method->setSchema(schema);
    return 0;    
}

torch::jit::Module build_tmp_module(std::shared_ptr<torch::jit::Graph>& sub_graph) {
    torch::jit::script::Module new_mod("tmp_submodule");
    auto graph = sub_graph->copy();
    merge_graph_to_module(graph, new_mod, true);
    return new_mod;
}

//好好参考aten/src/ATen/core/type.cpp 里 operator<< 的写法
bool gen_dims_for_tensor(const torch::jit::Value* value, std::vector<int64_t>& dims) {
    POROS_CHECK_TRUE((value->type()->isSubtypeOf(c10::TensorType::get())), 
        "given value for gen_dims_for_tensor is not Tensor as expected");
    std::vector<int64_t>().swap(dims);
    c10::TensorTypePtr op = value->type()->cast<c10::TensorType>();
    if (auto ndim = op->sizes().size()) {
        for (size_t i = 0; i < *ndim; ++i) {
            if (auto s = op->sizes()[i]) {
                dims.push_back(s.value());
            } else {
                dims.push_back(-1);
            }
        }
        return true;
    }
    return false;
}

void update_global_context(torch::jit::Value* old_value, torch::jit::Value* new_value) {
    // copy value_dynamic_shape_map
    if (old_value->type()->isSubtypeOf(c10::TensorType::get())) {
        if (PorosGlobalContext::instance()._value_dynamic_shape_map.count(old_value) > 0) {
            PorosGlobalContext::instance()._value_dynamic_shape_map[new_value] =
                PorosGlobalContext::instance()._value_dynamic_shape_map[old_value];
        }
    } else if (old_value->type()->kind() == c10::TypeKind::IntType) {
        update_global_int_intlist_map_context(old_value, new_value);
    } else if (old_value->type()->kind() == c10::TypeKind::ListType) {
        if (old_value->type()->isSubtypeOf(c10::ListType::ofInts())) {
            update_global_int_intlist_map_context(old_value, new_value);
        }
        PorosGlobalContext::instance()._list_size_map.update_value(old_value, new_value);   
    } else {

    }
    //to add more @wangrui39
}

void update_global_int_intlist_map_context(torch::jit::Value* old_value, torch::jit::Value* new_value) {
    if (PorosGlobalContext::instance()._int_intlist_values_map.count(old_value) > 0) {
            PorosGlobalContext::instance()._int_intlist_values_map[new_value] =
                PorosGlobalContext::instance()._int_intlist_values_map[old_value];
    }
}

void update_global_list_size_map_node_key_context(torch::jit::Node* old_node, torch::jit::Node* new_node) {
    PorosGlobalContext::instance()._list_size_map.update_node(old_node, new_node);   
}

void unmerge_subgraph(torch::jit::Node* subgraph_node) {
    // Inline the graph, replace uses of node outputs and destroy the node
    auto outer_graph = subgraph_node->owningGraph();
    std::shared_ptr<torch::jit::Graph> sub_graph = subgraph_node->g(torch::jit::attr::Subgraph);

    torch::jit::WithInsertPoint guard(subgraph_node);
    const auto subgraph_outputs = torch::jit::insertGraph(
        *outer_graph, *sub_graph, subgraph_node->inputs());
    AT_ASSERT(subgraph_outputs.size() >= subgraph_node->outputs().size());
    for (size_t i = 0; i < subgraph_node->outputs().size(); ++i) {
        subgraph_node->outputs()[i]->replaceAllUsesWith(subgraph_outputs[i]);
        update_global_context(subgraph_node->outputs()[i], subgraph_outputs[i]);
    }
    subgraph_node->destroy();
}

void find_to_optimized_nodes(torch::jit::Block* block, std::vector<torch::jit::Node*>& to_optimized_nodes) {
    //bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
        torch::jit::Node* node = *it;
        for (torch::jit::Block* subblock : node->blocks()) {
            find_to_optimized_nodes(subblock, to_optimized_nodes);
        }
        if (node->kind() == torch::jit::prim::CudaFusionGroup) {
            to_optimized_nodes.push_back(node);
        }
    }
}


/********************************************************************
             SOME DEPRECATED FUNCTIONS BELOW
*********************************************************************/
//DEPRECATED
bool gen_dims_for_scarlar(const torch::jit::Value* value, std::vector<int64_t>& dims) {
   return false; 
}

// gen dims for tensorlist input
// DEPRECATED
bool gen_dims_for_tensorlist(const torch::jit::Value* value, std::vector<int64_t>& dims) {
    // if we want to treat the tensorlist as a single input to tensort. 
    // TODO: we should check the tensors in list are of the same size. 
    // std::vector<int64_t> pre_dims;
    POROS_CHECK_TRUE(value->type()->isSubtypeOf(c10::ListType::ofTensors()), 
        "given value for gen_dims_for_tensorlist is not TensorList as expected");
    std::vector<int64_t>().swap(dims);

    auto producer = value->node();
    if (producer->kind() == torch::jit::aten::meshgrid) {
        LOG(INFO) << "to support: torch::jit::aten::meshgrid";
    
    // prim::ListConstruct
    } else if (producer->kind() == torch::jit::prim::ListConstruct) {
        //situation one: some node like : 
        // %out : Tensor[] = prim::ListConstruct(%intput)
        if (producer->inputs().size() > 0) {
            auto op = producer->inputs()[0]->type()->cast<c10::TensorType>();
            if (op->sizes().size().has_value() && op->scalarType().has_value()) {
                dims = op->sizes().concrete_sizes().value();
                return true;
            }
        //situation two: some node like: 
        //  %out : Tensor[] = prim::ListConstruct()
        //  %new_out: Tensor[] = aten::append(%out, %item)
        } else {
            for (auto use: value->uses()) {
                LOG(INFO) << "checking user: " << node_info_with_attr(use.user);
                if(use.user->kind() == torch::jit::aten::append &&
                    use.user->inputs().size() > 1) {
                    auto op = use.user->inputs()[1]->type()->cast<c10::TensorType>();
                    if (op->sizes().size().has_value() && op->scalarType().has_value()) {
                        dims = op->sizes().concrete_sizes().value();
                        return true;
                    }
                }
            }
        }
        LOG(INFO) <<  "to support: torch::jit::prim::ListConstruct";
    } else if (producer->kind() == torch::jit::prim::Constant) {
        LOG(INFO) <<  "to support: torch::jit::prim::Constant";
    } else {
        // aten::unbind
        // prim::Constant
        LOG(INFO) <<  "to support: some kind of producer: "  << producer->kind().toQualString();
    }
    return false;
}

//TODO: this method is relatively low-level, try to use SubgraphRewriter to handle this one
//DEPRECATED
bool is_linear_if_node(torch::jit::Node* node) {
    /// Check if this Node hosts a pattern like so:
    ///  %ret = prim::If(%1)
    ///      block0():
    ///          %ret1 = aten::addmm(%bias, %input, %weight_t, %beta, %alpha)
    ///          -> (%ret1)
    ///      block1():
    ///          %output = aten::matmul(%input, %weight_t)
    ///          %ret2 = aten::add(%output, %bias, %alpha)
    ///          -> (%ret2)
    if (node->kind() != torch::jit::prim::If || node->blocks().size() != 2) {
        return false;
    }

    auto block2vector = [](torch::jit::Block* block, std::vector<torch::jit::Node*>& nodes_vec) {
        for (auto itr : block->nodes()) {
            nodes_vec.emplace_back(itr);
        }
    };

    std::vector<torch::jit::Node*> true_nodes;
    std::vector<torch::jit::Node*> false_nodes;
    block2vector(node->blocks()[0], true_nodes);
    block2vector(node->blocks()[1], false_nodes);

    if (node->blocks()[0]->outputs().size() != 1 || 
        true_nodes.size() != 1 ||
        true_nodes[0]->kind() != torch::jit::aten::addmm) {
            return false;
    }

    if (node->blocks()[1]->outputs().size() != 1 || 
        false_nodes.size() != 2 || 
        false_nodes[0]->kind() != torch::jit::aten::matmul ||
        false_nodes[1]->kind() != torch::jit::aten::add ) {
        return false;
    }
    
    auto is_input_const = [](torch::jit::Node* node, int index) {
        return (node->inputs()[index])->node()->kind() == torch::jit::prim::Constant;
    };

    if (true_nodes[0]->inputs().size() != 5 ||
        !is_input_const(true_nodes[0], 0) ||
        !is_input_const(true_nodes[0], 2) || 
        !is_input_const(true_nodes[0], 3) || 
        !is_input_const(true_nodes[0], 4)) {
        return false;
    }

    if (false_nodes[0]->inputs().size() != 2 ||
        !is_input_const(false_nodes[0], 1) ||
        false_nodes[0]->inputs()[0] != true_nodes[0]->inputs()[1] ||
        false_nodes[0]->inputs()[1] != true_nodes[0]->inputs()[2] ||
        false_nodes[1]->inputs()[0] != false_nodes[0]->outputs()[0] ||
        false_nodes[1]->inputs()[1] != true_nodes[0]->inputs()[0] ||
        false_nodes[1]->inputs()[2] != true_nodes[0]->inputs()[4]) {
        return false;
    }

    return true;
}

//DEPRECATED
std::vector<torch::jit::Value*> extract_linear_input(torch::jit::Node *node) {
    std::vector<torch::jit::Value*> valid_inputs;
    if (is_linear_if_node(node)) {
        valid_inputs.emplace_back(node->inputs()[0]);
        auto addmm_node = *((node->blocks()[0])->nodes().begin());
        valid_inputs.emplace_back(addmm_node->inputs()[1]);
    }
    return valid_inputs;
}

//DEPRECATED
bool is_dim_equal_if_node(torch::jit::Node* node) {
    /// Check if this Node hosts a pattern like so:
    /// %const_val : int = prim::Constant[value=2]()
    /// %dim : int = aten::dim(%input_tensor)
    /// %eq : bool = aten::eq(%dim, %const_val)
    /// %ret = prim::If(%eq)
    ///     block0():
    ///     ...
    ///     block1():
    ///     ...
    if (node->kind() != torch::jit::prim::If || node->blocks().size() != 2 ||
        node->inputs().size() != 1) {
        return false;
    }

    if (node->input(0)->node()->kind() == torch::jit::aten::eq) {
        auto eq_node = node->input(0)->node();
        if (eq_node->inputs().size() == 2 &&
            eq_node->input(0)->node()->kind() == torch::jit::aten::dim &&
            eq_node->input(1)->node()->kind() == torch::jit::prim::Constant) {
            return true;
        }
    }
    return false;
}

//DEPRECATED
void inline_if_body(torch::jit::Block* body) {
    torch::jit::Node* owning_node = body->owningNode();
    for (auto itr = body->nodes().begin(); itr != body->nodes().end();) {
        torch::jit::Node* body_node = *itr;
        // advance iterator because after body_node is moved its next pointer will be to n
        itr++;
        body_node->moveBefore(owning_node);
    }
    for (size_t i = 0; i < owning_node->outputs().size(); ++i) {
        owning_node->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
    }
    owning_node->destroy();
}

/*
  bool shapeIsKnown(Value* v) {
    if (v->type()->cast<TensorType>()) {
      if (!v->isCompleteTensor()) {
        return false;
      }
      if (*v->type()->castRaw<TensorType>()->dim() == 0) {
        return false;
      }
    }
    return true;
  } */

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
