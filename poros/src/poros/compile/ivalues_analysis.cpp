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
* @file ivalues_analysis.cpp
* @author tianjinjin@baidu.com
* @date Fri Apr 23 11:41:59 CST 2021
* @brief
**/
#include "poros/compile/ivalues_analysis.h"

#include <sstream>

#include <c10/util/Exception.h>
#include <torch/script.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>  //check
#include <torch/csrc/jit/runtime/interpreter.h>  //to get executioncontext
#include <torch/csrc/jit/runtime/graph_executor.h>  //for get numprofileruns

#include "poros/util/poros_util.h"
#include "poros/context/poros_global.h"

//I have to copy the ProfileOp function here
namespace torch {
namespace jit {

const Symbol ProfileOp::Kind = ::c10::prim::profile;
void ProfileOp::cloneFrom(torch::jit::Node* other_) {
    torch::jit::Node::cloneFrom(other_);
    auto other = other_->cast<ProfileOp>();
    this->callback_ = other->getCallback();
}

torch::jit::Node* ProfileOp::allocNewInstance(torch::jit::Graph* g) {
    return new ProfileOp(g, {nullptr});
}

}  // namespace jit
}  // namespace torch

namespace baidu {
namespace mirana {
namespace poros {

IvalueAnalysis::IvalueAnalysis(std::shared_ptr<torch::jit::Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(torch::jit::getNumProfiledRuns()) {}

void IvalueAnalysis::insert_input_listsize_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* input_value = node->input(offset);

    // 创建profile node
    torch::jit::ProfileOp *pn = create_profile_node(nullptr, {input_value});
    auto pno = pn->addOutput();
    pn->ty_(torch::jit::attr::profiled_type, input_value->type());
    pno->setType(input_value->type());

    std::function<void(torch::jit::Stack&)> list_size_profile = [this, pno, node](torch::jit::Stack& stack){
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (ivalue.isList()) {
            auto input_value = pno->node()->input(0);
            //not exist yet, insert it
            if (_list_size_map._list_size_map_input.count(input_value) == 0) {
                std::set<int32_t> int_set{(int32_t)ivalue.toListRef().size()};
                _list_size_map._list_size_map_input[input_value][node] = int_set;
                if (ivalue.isTensorList()) {
                    auto tl = ivalue.toTensorList();
                    std::map<int32_t, std::vector<c10::TensorTypePtr>> type_map;
                    for(size_t i = 0; i < ivalue.toListRef().size(); i++){
                        auto tlty = torch::jit::tensorTypeInCurrentExecutionContext(tl[i]);
                        type_map[i] = {tlty};
                    }
                    _list_size_map._list_tensor_type_map_input[input_value][node] = type_map;
                }
            }
            else {
                if (_list_size_map._list_size_map_input[input_value].count(node) == 0) {
                    std::set<int32_t> int_set{(int32_t)ivalue.toListRef().size()};
                    _list_size_map._list_size_map_input[input_value][node] = int_set;
                }
                else {
                    _list_size_map._list_size_map_input[input_value][node].insert(ivalue.toListRef().size());
                }
                if (ivalue.isTensorList()) {
                    auto tl = ivalue.toTensorList();
                    std::map<int32_t, std::vector<c10::TensorTypePtr>> &type_map = _list_size_map._list_tensor_type_map_input[input_value][node];
                    for(size_t i = 0; i < ivalue.toListRef().size(); i++) {
                        auto tlty = torch::jit::tensorTypeInCurrentExecutionContext(tl[i]);
                        type_map[i].push_back(tlty);
                    }
                }
            }
            // extract int[] values to map
            if (input_value->type()->isSubtypeOf(c10::ListType::ofInts()) && 
                input_value->node()->kind() != torch::jit::prim::Constant && ivalue.isIntList()) {
                auto& value_vec_map = _int_intlist_values_per_frame[frame_id];
                // extract int[]
                std::vector<int64_t> int_vec;
                c10::List<int64_t> c10_int_list = ivalue.toIntList();
                for (int64_t i : c10_int_list) {
                    int_vec.push_back(i);
                }
                //not exist yet, insert it
                if (value_vec_map.count(input_value) == 0) {
                    std::vector<std::vector<int64_t>> int_vec_vec;
                    int_vec_vec.push_back(int_vec);
                    value_vec_map.insert({input_value, int_vec_vec});
                } else {
                    value_vec_map[input_value].push_back(int_vec);
                }
            }    
        }
        torch::jit::push(stack, ivalue);
    };
    pn->setCallback(list_size_profile);
    pn->insertBefore(node);
    node->replaceInput(offset, pn->output());
}

void IvalueAnalysis::insert_number_eval_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* input_value = node->input(offset);
    // 创建profile node
    torch::jit::ProfileOp *pn = create_profile_node(nullptr, {input_value});
    auto pno = pn->addOutput();
    pn->ty_(torch::jit::attr::profiled_type, input_value->type());
    pno->setType(input_value->type());
    
    std::function<void(torch::jit::Stack&)> int_intlist_profile = [this, input_value](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (ivalue.isInt()) {
            auto& value_vec_map = _int_intlist_values_per_frame[frame_id];
            // extract int
            std::vector<int64_t> int_vec;
            int_vec.push_back(ivalue.toInt());
            //not exist yet, insert it
            if (value_vec_map.count(input_value) == 0) {
                std::vector<std::vector<int64_t>> int_vec_vec;
                int_vec_vec.push_back(int_vec);
                value_vec_map.insert({input_value, int_vec_vec});
            } else {
                value_vec_map[input_value].push_back(int_vec);
            }
        }
        // passing t through
        torch::jit::push(stack, ivalue);
    };
    pn->setCallback(int_intlist_profile);
    pn->insertBefore(node);
    node->replaceInput(offset, pn->output());
}

void IvalueAnalysis::insert_output_listsize_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* output_value = node->output(offset);
    //watch this value
    auto eval_pn = create_profile_node(nullptr, {output_value});
    auto pno = eval_pn->addOutput();
    eval_pn->ty_(torch::jit::attr::profiled_type, output_value->type());
    pno->setType(output_value->type());

    //do we need outout? and change the input of prim::If?
    std::function<void(torch::jit::Stack&)> eval_profiler = [this, pno, node](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        std::lock_guard<std::mutex> lock(this->mutex_);

        if (ivalue.isList()) {
            //not exist yet, insert it
            auto input_value = pno->node()->input(0);
            if (_list_size_map._list_size_map_output.count(input_value) == 0) {
                std::set<int32_t> int_set{(int32_t)ivalue.toListRef().size()};
                _list_size_map._list_size_map_output[input_value][node] = int_set;

                if (ivalue.isTensorList()) {
                    auto tl = ivalue.toTensorList();
                    std::map<int32_t, std::vector<c10::TensorTypePtr>> type_map;
                    for(size_t i = 0; i < ivalue.toListRef().size(); i++){
                        auto tlty = torch::jit::tensorTypeInCurrentExecutionContext(tl[i]);
                        type_map[i] = {tlty};
                    }
                    _list_size_map._list_tensor_type_map_output[input_value][node] = type_map;
                }
            }
            else {
                if (_list_size_map._list_size_map_output[input_value].count(node) == 0) {
                    std::set<int32_t> int_set{(int32_t)ivalue.toListRef().size()};
                    _list_size_map._list_size_map_output[input_value][node] = int_set;
                }
                else {
                    _list_size_map._list_size_map_output[input_value][node].insert(ivalue.toListRef().size());
                }

                if (ivalue.isTensorList()) {
                    auto tl = ivalue.toTensorList();
                    std::map<int32_t, std::vector<c10::TensorTypePtr>> &type_map = _list_size_map._list_tensor_type_map_output[input_value][node];
                    for(size_t i = 0; i < ivalue.toListRef().size(); i++) {
                        auto tlty = torch::jit::tensorTypeInCurrentExecutionContext(tl[i]);
                        type_map[i].push_back(tlty);
                    }
                }
            }
        }
        torch::jit::push(stack, ivalue);
    };
    eval_pn->setCallback(eval_profiler);
    eval_pn->insertAfter(node);
}

//TODO: check out the difference between the ProfileIValueOp and ProfileOp
torch::jit::ProfileOp* IvalueAnalysis::create_profile_node(
                                const std::function<void(torch::jit::Stack&)>& fp,
                                at::ArrayRef<torch::jit::Value*> inputs) {
    auto pn = new torch::jit::ProfileOp(profiled_graph_.get(), fp);
    for (auto in : inputs) {
        pn->addInput(in);
    }
    return pn;
}

/*
torch::jit::ProfileOptionalOp* IvalueAnalysis::create_profile_optional_node(
                            const std::function<void(torch::jit::Stack&)>& fp,
                            at::ArrayRef<torch::jit::Value*> inputs) {
    auto pn = new torch::jit::ProfileOptionalOp(profiled_graph_.get(), fp);
    pn->i_(torch::jit::attr::num_present, 0);
    pn->i_(torch::jit::attr::num_none, 0);
    for (auto in : inputs) {
        pn->addInput(in);
    }
    return pn;
} */

void IvalueAnalysis::insert_shape_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* input_value = node->input(offset);
    //watch this value
    auto pn = create_profile_node(nullptr, {input_value});
    auto pno = pn->addOutput();
    pn->ty_(torch::jit::attr::profiled_type, c10::TensorType::get());
    pno->setType(c10::TensorType::get());
    
    std::function<void(torch::jit::Stack&)> shape_profiler = [this, pno](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        if (ivalue.isTensor()) {
            std::lock_guard<std::mutex> lock(this->mutex_);
            std::map<torch::jit::Value*, std::vector<c10::TensorTypePtr>>& profiled_types = _profiled_types_per_frame[frame_id];
            at::Tensor t = ivalue.toTensor();
            if (t.defined()) {
                at::TensorTypePtr pttp = torch::jit::tensorTypeInCurrentExecutionContext(t);
                if (profiled_types.count(pno) == 0) {
                    //insert value and tensortype info
                    profiled_types.insert({pno, {pttp}});
                } else {
                    std::vector<c10::TensorTypePtr>& type_list = profiled_types.at(pno);
                    type_list.push_back(pttp);
                    // auto type = profiled_types.at(pno);
                    // pttp = type->merge(*pttp);
                    // profiled_types[pno] = pttp;
                }
            } else {
                profiled_types[pno] = {c10::TensorType::get()->withUndefined()};
            }
        }
        // passing t through
        torch::jit::push(stack, ivalue);
    };
    pn->setCallback(shape_profiler);
    pn->insertBefore(node);
    node->replaceInput(offset, pn->output());
}

/*
void IvalueAnalysis::insert_optional_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* input_value = node->input(offset);
    // this value
    auto opt_pn = create_profile_optional_node(nullptr, {input_value});
    // watch the definition instead of the use, 
    // because we are only optimizing in the case of a None value which is immutable
    std::function<void(torch::jit::Stack&)> optional_profiler = [this, opt_pn](torch::jit::Stack& stack) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        if (ivalue.isNone()) {
            opt_pn->i_(torch::jit::attr::num_none, opt_pn->i(torch::jit::attr::num_none) + 1);
        } else {
            opt_pn->i_(torch::jit::attr::num_present, opt_pn->i(torch::jit::attr::num_present) + 1);
        }
        torch::jit::push(stack, ivalue);
    };
    opt_pn->setCallback(optional_profiler);
    auto pno = opt_pn->addOutput();
    pno->setType(input_value->type());
    opt_pn->insertAfter(input_value->node());
    input_value->replaceAllUsesAfterNodeWith(opt_pn, pno);
} */

//TODO: check more
void IvalueAnalysis::insert_eval_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* output_value = node->output(offset);
    //watch this value
    auto eval_pn = create_profile_node(nullptr, {output_value});
    auto pno = eval_pn->addOutput();
    eval_pn->ty_(torch::jit::attr::profiled_type, output_value->type());
    pno->setType(output_value->type());

    //do we need outout? and change the input of prim::If?
    std::function<void(torch::jit::Stack&)> eval_profiler = [this, pno](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (ivalue.isBool()) {
            //not exist yet, insert it
            if (_evaluate_values_map.count(pno) == 0) {
                std::vector<bool> bool_vector{ivalue.toBool()};
                _evaluate_values_map[pno] = bool_vector;
            } else {
                _evaluate_values_map[pno].emplace_back(ivalue.toBool());
            }
        }
        torch::jit::push(stack, ivalue);
    };
    eval_pn->setCallback(eval_profiler);
    eval_pn->insertAfter(node);
    //replace the user nodes.
    for (auto use: output_value->uses()) {
        auto consumer_node = use.user;
        for(size_t offset = 0; offset < consumer_node->inputs().size(); offset++) {
            if (consumer_node->input(offset) == output_value &&
                consumer_node != eval_pn) {
                consumer_node->replaceInput(offset, eval_pn->output());
            }
        }
    }
}

std::map<torch::jit::Value*, c10::TensorTypePtr> IvalueAnalysis::merge_tensor_type_per_frame(
                    std::map<torch::jit::Value*, std::vector<c10::TensorTypePtr>>& profiled_map) {
    std::map<torch::jit::Value*, c10::TensorTypePtr> merged_tensor_type;
    for (auto iter : profiled_map) {
        torch::jit::Value* profile_value = iter.first;
        std::vector<c10::TensorTypePtr> type_list = iter.second;
        for (auto tensor_type : type_list) {
            if (merged_tensor_type.count(profile_value) == 0) {
                merged_tensor_type.insert({profile_value, tensor_type});
            } else {
                c10::TensorTypePtr type = merged_tensor_type.at(profile_value);
                tensor_type = type->merge(*tensor_type);
                merged_tensor_type[profile_value] = tensor_type;
            }
        }
    }
    return merged_tensor_type;
}

c10::SymbolicShape IvalueAnalysis::merge_symbolic_shapes(
                                const c10::SymbolicShape& new_sizes,
                                const c10::SymbolicShape& sym_shapes,
                                torch::jit::SetPartitioningHelper& partition_helper) {
    std::vector<c10::ShapeSymbol> new_symbols;
    TORCH_INTERNAL_ASSERT(
        new_sizes.rank().has_value() && sym_shapes.rank().has_value() &&
        *new_sizes.rank() == *sym_shapes.rank());

    for (size_t i = 0; i < *new_sizes.rank(); i++) {
        if (!(*sym_shapes.sizes())[i].is_static() ||
            !(*new_sizes.sizes())[i].is_static()) {
            new_symbols.emplace_back();
            continue;
        }
        auto symbol = (*sym_shapes.sizes())[i];
        int64_t new_size = (*new_sizes.sizes())[i].static_size();
        //GRAPH_DUMP("Merging symbol ", symbol);
        auto new_sym = partition_helper.partitionSetByDimension(new_size, symbol);
        new_symbols.emplace_back(new_sym);
    }
    return c10::SymbolicShape(new_symbols);
}

void IvalueAnalysis::analysis_ivalue_for_block(torch::jit::Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
        auto node = *it;
        //iterate the input value of the node
        for (size_t offset = 0; offset < node->inputs().size(); offset++) {
            auto input_value = node->input(offset);
            //tensortype handle
            if (input_value->type()->kind() == c10::TypeKind::TensorType) {
                insert_shape_profile(node, offset);
            }
            if (input_value->type()->kind() == c10::TypeKind::ListType){
                insert_input_listsize_profile(node, offset);
            }
            // 踩坑记录：0318，须保证后面分析的value没有被前面加过profile node
            // 否则前面profile output替换了value节点，map key中找不到自己想要的value
            // 且同一value的profile callback函数会多次执行，造成不可预知的问题
            if (input_value->type()->kind() == c10::TypeKind::IntType && 
                input_value->node()->kind() != torch::jit::prim::Constant) {
                insert_number_eval_profile(node, offset);
            }
            //TODO: WHY NOT SUPPORT ProfileOptionalOp anymore after I upgrade libtorch from 1.7.1 to 1.8.1
            // if (input_value->type()->cast<c10::OptionalType>() && 
            //     has_gradsum_to_size_uses(input_value)) {
            //     insert_optional_profile(node, offset);
            // }

            if (input_value->type()->kind() == c10::TypeKind::BoolType) {
                insert_eval_profile(input_value->node(), 0);
                //TODO: modify the second input 0 to more strict check
            }
        }
        for (size_t offset = 0; offset < node->outputs().size(); offset++) {
            auto output_value = node->output(offset);
            if (output_value->type()->kind() == c10::TypeKind::ListType){
                insert_output_listsize_profile(node, offset);
                it++;
            }
        }

        for (auto b : node->blocks()) {
            analysis_ivalue_for_block(b);
        }
    }
    
    //insert shape profile for block outputs
    for (size_t offset = 0; offset < block->return_node()->inputs().size(); offset++) {
        auto input_value = block->return_node()->input(offset);
        if (input_value->type()->isSubtypeOf(c10::TensorType::get())) {
            insert_shape_profile(block->return_node(), offset);
        }
        // //TODO: should I add this??
        // if (input_value->type()->kind() == c10::TypeKind::BoolType) {
        //     insert_eval_profile(input_value->node(), 0);
        // }
    }
}

void IvalueAnalysis::gen_list_size() {
    PorosGlobalContext::instance()._list_size_map = _list_size_map;
}

void IvalueAnalysis::gen_value_dyanamic_shape() {

    std::map<torch::jit::Value*, ValueDynamicShape>& value_dynamic_shape_map = PorosGlobalContext::instance()._value_dynamic_shape_map;
    if (_profiled_types_per_frame.size() < 3) {
        throw c10::Error("dynamic_shape must has three prewarm data [max & min & opt]", "");
    }

    auto profiled_types_iter = _profiled_types_per_frame.begin();  //frame id 
    auto start_frame_id = profiled_types_iter->first;   // std::map<torch::jit::Value*, c10:TensorTypePtr>

    //max
    for (auto &e : _profiled_types_per_frame[start_frame_id++]) {
        auto profile_value = e.first->node()->input();
        //没有则创建。
        if (value_dynamic_shape_map.count(profile_value) == 0) {
            ValueDynamicShape shape;
            value_dynamic_shape_map[profile_value] = shape;
            value_dynamic_shape_map[profile_value].is_dynamic = false;
        }

        std::vector<c10::TensorTypePtr>& shape_list = e.second;
        std::vector<int64_t> current_shape;
        for (auto &shape :  shape_list) {
            if (shape->sizes().concrete_sizes().has_value()) {
                current_shape = shape->sizes().concrete_sizes().value();
            } else {
                // 因为此处在剪枝操作之前，有的block可能没有被执行。
                // 而这些block中的profile node是空值，此处应跳过
                continue;
            }
            //当一个value 作为多个node的输入的时候，会有多个profile。
            //其次，当一个tensor出现在loop中的时候，相关联的op大概率会被多次执行，也会有多个profile。
            //2021.11.11 踩坑记录，针对多个profile，如果出现了size 不一致的情况，max 应该取其中最大的，min应该取其中最小的。
            if (value_dynamic_shape_map[profile_value].max_shapes.size() != 0) {
                auto old_shape = value_dynamic_shape_map[profile_value].max_shapes;
                std::vector<int64_t> new_shape;
                for (size_t i = 0; i < old_shape.size(); ++i) {
                    new_shape.push_back(std::max(old_shape[i], current_shape[i]));
                }
                value_dynamic_shape_map[profile_value].max_shapes = new_shape;
                // LOG(INFO) << "try to update max shape, current_shape: [" << current_shape
                //           << "], old_shape: [" << old_shape
                //           << "], new_shape: [" << new_shape << "]";
            } else {
                value_dynamic_shape_map[profile_value].max_shapes = current_shape;
            }
        }
    }

    //min
    for (auto &e : _profiled_types_per_frame[start_frame_id++]) {
        //TODO: maybe need to check the value existing before setting
        auto profile_value = e.first->node()->input();
        std::vector<c10::TensorTypePtr> shape_list = e.second;
        std::vector<int64_t> current_shape;
        for (auto &shape :  shape_list) {
            if (shape->sizes().concrete_sizes().has_value()) {
                current_shape = shape->sizes().concrete_sizes().value();
            } else {
                // 因为此处在剪枝操作之前，有的block可能没有被执行。
                // 而这些block中的profile node是空值，此处应跳过
                continue;
            }
            if (value_dynamic_shape_map[profile_value].min_shapes.size() != 0) {
                auto old_shape = value_dynamic_shape_map[profile_value].min_shapes;
                std::vector<int64_t> new_shape;
                for (size_t i = 0; i < old_shape.size(); ++i) {
                    new_shape.push_back(std::min(old_shape[i], current_shape[i]));
                }
                value_dynamic_shape_map[profile_value].min_shapes = new_shape;
                // LOG(INFO) << "try to update min shape, current_shape: [" << current_shape
                //           << "], old_shape: [" << old_shape
                //           << "], new_shape: [" << new_shape << "]";
            } else {
                value_dynamic_shape_map[profile_value].min_shapes = current_shape;
            }
        }
    }

    //opt
    for (auto &e : _profiled_types_per_frame[start_frame_id++]) {
        auto profile_value = e.first->node()->input();
        std::vector<c10::TensorTypePtr> shape_list = e.second;
        for (auto &shape :  shape_list) {
            if (shape->sizes().concrete_sizes().has_value()) {
                value_dynamic_shape_map[profile_value].opt_shapes = shape->sizes().concrete_sizes().value();
            }
        }
    }

    for (auto &e : value_dynamic_shape_map) {
        ValueDynamicShape& shape = e.second;
        //2022.09.28 踩坑记录，当针对某一个value, 出现了其中一个shape的size为0的情况，
        //说明这个value在某个block下(可能是循环次数跟query相关的loop，也可能是进入条件跟query相关的if分支)
        //此时对该block下的graph进行子图分割会出现异常，因为在子图替换阶段，无法正常生产输入的size信息。
        //此处兼容这种情况。
        if (shape.max_shapes.size() == 0 || shape.min_shapes.size() == 0 || shape.opt_shapes.size() == 0) {
            if (e.first->node()->kind() == torch::jit::prim::Constant) {
                continue;
            }
            LOG(INFO) << "value shape info for: %" << e.first->debugName()
                    << ", max_shape: " << shape.max_shapes
                    << ", min_shape: " << shape.min_shapes
                    << ", opt_shape: " << shape.opt_shapes;       
            PorosGlobalContext::instance()._disable_subblock_convert = true;
            continue;
        }
        for (size_t i = 0; i < shape.max_shapes.size(); ++i) {
            if (shape.max_shapes[i] == shape.min_shapes[i] && shape.max_shapes[i] == shape.opt_shapes[i]) {
                shape.sizes.push_back(shape.max_shapes[i]); 
            } else {
                shape.sizes.push_back(-1);
                shape.is_dynamic = true;
            }
        }
        // LOG(INFO) << "value shape info for: %" << e.first->debugName()
        //          << ", max_shape: " << shape.max_shapes
        //          << ", min_shape: " << shape.min_shapes
        //          << ", opt_shape: " << shape.opt_shapes;
    }
}

void IvalueAnalysis::gen_int_intlist_value() {
    std::map<torch::jit::Value*, ValueDynamicShape>& int_intlist_values_map = PorosGlobalContext::instance()._int_intlist_values_map;
    if (_int_intlist_values_per_frame.size() == 0) {
        return;
    }

    int64_t start_frame_id = _int_intlist_values_per_frame.begin()->first;  //frame id
    
    //max
    // e -> std::map<torch::jit::Value*, std::vector<std::vector<int64_t>>> 迭代 pair
    for (auto &e : _int_intlist_values_per_frame[start_frame_id++]) {
        std::vector<std::vector<int64_t>> int_values_vecs = e.second;
        if (int_values_vecs.size() == 0) {
            continue;
        }
        size_t per_vec_size = int_values_vecs[0].size();
        std::vector<int64_t> max_vector(per_vec_size, INT64_MIN);
        bool length_is_var = false;
        for (std::vector<int64_t>& v : int_values_vecs) {
            if (max_vector.size() != v.size()) {
                length_is_var = true;
                break;
            }
            for (size_t i = 0; i < max_vector.size(); i++) {
                max_vector[i] = std::max(max_vector[i], v[i]);
            }
        }
        if (length_is_var) {
            continue;
        }

        torch::jit::Value* int_value = e.first;
        if (int_intlist_values_map.count(int_value) == 0) {
            ValueDynamicShape shape;
            shape.max_shapes = max_vector;
            int_intlist_values_map[int_value] = shape;
            int_intlist_values_map[int_value].is_dynamic = false;
        } else {
            int_intlist_values_map[int_value].max_shapes = max_vector;
        }
    }

    if (_int_intlist_values_per_frame.size() == 1) {
        for (auto &e : int_intlist_values_map) {
            e.second.min_shapes = e.second.max_shapes;
            e.second.opt_shapes = e.second.max_shapes;
        }
        return;
    } else {
        for (auto &e : _int_intlist_values_per_frame[start_frame_id++]) {
            std::vector<std::vector<int64_t>> int_values_vecs = e.second;
            if (int_values_vecs.size() == 0) {
                continue;
            }
            size_t per_vec_size = int_values_vecs[0].size();
            std::vector<int64_t> min_vector(per_vec_size, INT64_MAX);
            bool length_is_var = false;
            for (std::vector<int64_t>& v : int_values_vecs) {
                if (min_vector.size() != v.size()) {
                    length_is_var = true;
                    break;
                }
                for (size_t i = 0; i < min_vector.size(); i++) {
                    min_vector[i] = std::min(min_vector[i], v[i]);
                }
            }
            if (length_is_var) {
                continue;
            }

            torch::jit::Value* int_value = e.first;
            if (int_intlist_values_map.count(int_value) == 0) {
                ValueDynamicShape shape;
                shape.min_shapes = min_vector;
                int_intlist_values_map[int_value] = shape;
                int_intlist_values_map[int_value].is_dynamic = false;
            } else {
                int_intlist_values_map[int_value].min_shapes = min_vector;
            }
        }

        //opt
        for (auto &e : _int_intlist_values_per_frame[start_frame_id++]) {
            std::vector<std::vector<int64_t>> int_values_vecs = e.second;
            if (int_values_vecs.size() == 0 ) {
                continue;
            }
            size_t per_vec_size = int_values_vecs[0].size();
            bool length_is_var = false;
            for (std::vector<int64_t>& v : int_values_vecs) {
                if (per_vec_size != v.size()) {
                    length_is_var = true;
                    break;
                }
            }
            if (length_is_var) {
                continue;
            }

            torch::jit::Value* int_value = e.first;
            if (int_intlist_values_map.count(int_value) == 0) {
                ValueDynamicShape shape;
                shape.opt_shapes = int_values_vecs[0];
                int_intlist_values_map[int_value] = shape;
                int_intlist_values_map[int_value].is_dynamic = false;
            } else {
                int_intlist_values_map[int_value].opt_shapes = int_values_vecs[0];
            }
        }
    }
}

std::unique_ptr<IvalueAnalysis> IvalueAnalysis::analysis_ivalue_for_graph(
                    const std::shared_ptr<torch::jit::Graph>& graph) {

    auto new_g = graph->copy(); //copy or use the original one??
    auto ia = std::unique_ptr<IvalueAnalysis>(new IvalueAnalysis(new_g));
    auto raw_ia = ia.get();

    //clear the existing profile node that may exist.
    torch::jit::ClearProfilingInformation(new_g);
    //analysis main function
    ia->analysis_ivalue_for_block(new_g->block());
    
    std::function<void(torch::jit::Stack&)> counter = [raw_ia](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        std::lock_guard<std::mutex> lock(raw_ia->mutex_);
        
        if (raw_ia->profiling_count_ > 0) {
            raw_ia->profiling_count_--;
        }

        // merge tensortype profiling information from all runs
        if (raw_ia->profiling_count_ == 0) {
            LOG(INFO) << "Collected tensor profile " << raw_ia->_profiled_types_per_frame.size() << " records for run " << frame_id;
            if  (raw_ia->_profiled_types_per_frame.empty()) {
                return;
            }
            // the key is a frame id, the value is a mapping from a Value in a graph to a profiled TensorType
            // we make a copy of profiling information from the very first run
            // and use it for building the symbol sets
            auto profiled_types_iter = raw_ia->_profiled_types_per_frame.begin();  //frame id
            
            // merge itself
            auto merged_profiled_types = raw_ia->merge_tensor_type_per_frame(profiled_types_iter->second);
            ++profiled_types_iter;
            
            // merge profiling information from next runs into the first one
            for (; profiled_types_iter != raw_ia->_profiled_types_per_frame.end(); ++profiled_types_iter) {
                torch::jit::SetPartitioningHelper partition_helper;
                for (const auto& val_type_pair : raw_ia->merge_tensor_type_per_frame(profiled_types_iter->second)) {
                    auto insertion_result = merged_profiled_types.insert(val_type_pair);
                    if (!insertion_result.second) { // Already existed
                        const c10::TensorType* type = insertion_result.first->second.get();
                        //TODO: merge function take care more
                        //TODO: the merge function has change from torch1.7.1 to torch1.8.1
                        auto merged_type = type->merge(*val_type_pair.second);
                        if (merged_type->sizes().size().has_value()) {
                            auto new_shape = raw_ia->merge_symbolic_shapes(
                                val_type_pair.second->symbolic_sizes(), type->symbolic_sizes(), partition_helper);
                            GRAPH_DEBUG("Merging ", *val_type_pair.second, " of run ", profiled_types_iter->first, " into ", *type);
                            merged_type = type->withSymbolicShapes(std::move(new_shape));
                            GRAPH_DEBUG("Result : ", *merged_type);
                            insertion_result.first->second = std::move(merged_type);
                        } else {
                            // reset symbolic shapes when ranks are different
                            // TODO: attention here
                            insertion_result.first->second = std::move(merged_type);
                        }
                    }
                }
            }
            
            // update types in the graph
            for (auto val_type_pair : merged_profiled_types) {
                val_type_pair.first->node()->ty_(torch::jit::attr::profiled_type, val_type_pair.second);
            }
        }

        //TODO: check this more
        // update eval information from all runs
        if (raw_ia->profiling_count_ == 0) {
            LOG(INFO) << "Collected evaluate " << raw_ia->_evaluate_values_map.size() << " records for run " << frame_id;
            if  (raw_ia->_evaluate_values_map.empty()) {
                return;
            }

            torch::jit::WithInsertPoint guard(raw_ia->profiled_graph_->block()->nodes().front());
            auto true_const = raw_ia->profiled_graph_->insertConstant(true);
            auto false_const = raw_ia->profiled_graph_->insertConstant(false);
            for (auto& value_bools_pair : raw_ia->_evaluate_values_map) {
                auto profile_value = value_bools_pair.first;
                auto bool_vector = value_bools_pair.second;
                if (std::all_of(bool_vector.begin(), bool_vector.end(),
                    [](bool i){ return i == true;})) {
                    profile_value->node()->replaceInput(0, true_const);
                    //LOG(INFO) << "Replace " << node_info(profile_value->node()) << "input 0 as true_constant";
                }

                if (std::all_of(bool_vector.begin(), bool_vector.end(),
                    [](bool i){ return i == false;})) {
                    profile_value->node()->replaceInput(0, false_const);
                    //LOG(INFO) << "Replace " << node_info(profile_value->node()) << "input 0 as false_constant";
                }
            }
        }
    }; //func counter end
    
    auto pop = ia->create_profile_node(counter, {});
    new_g->appendNode(pop);  //put this profile at end of the graph to upback all the tensors.
    GRAPH_DUMP("Instrumented Graph: ", new_g);
    return ia;
}

//DEPRECATED
bool has_gradsum_to_size_uses(torch::jit::Value* v) {
    return std::any_of(v->uses().begin(), v->uses().end(), [](const torch::jit::Use& use) {
        return use.user->kind() == torch::jit::aten::_grad_sum_to_size;
    });
}

//DEPRECATED
void IvalueAnalysis::insert_debug_profile(torch::jit::Node* node, size_t offset) {
    torch::jit::Value* input_value = node->input(offset);
    //watch this value
    auto pn = create_profile_node(nullptr, {input_value});
    //auto pno = pn->addOutput();
    pn->ty_(torch::jit::attr::profiled_type, c10::TensorType::get());
    //pno->setType(c10::TensorType::get());

    std::function<void(torch::jit::Stack&)> debug_profiler = [this, node](torch::jit::Stack& stack) {
        int64_t frame_id = 0;
        torch::jit::pop(stack, frame_id);
        c10::IValue ivalue;
        torch::jit::pop(stack, ivalue);
        if (ivalue.isTensor()) {
            std::lock_guard<std::mutex> lock(this->mutex_);
            auto t = ivalue.toTensor();
            if (t.defined()) {
                auto pttp = torch::jit::tensorTypeInCurrentExecutionContext(t);

                //here. print node info.  print input info
                // std::cout << "debug during interprete, [node_info]:" << node_info_with_attr(node)
                //     <<", [input value type]: " << pttp->str()
                //     <<", [input value shape]: " << pttp->sizes().size()
                //     << std::endl;
            }
        }
        // passing t through
        torch::jit::push(stack, ivalue);
    };
    
    pn->setCallback(debug_profiler);
    pn->insertBefore(node);
    //node->replaceInput(offset, pn->output());
}

//DEPRECATED
void IvalueAnalysis::debug_tensors_for_block(torch::jit::Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
        auto node = *it;
        //iterate the input value of the node
        for (size_t offset = 0; offset < node->inputs().size(); offset++) {
            auto input_value = node->input(offset);
            //tensortype handle
            if (input_value->type()->kind() == c10::TypeKind::TensorType) {
                insert_debug_profile(node, offset);
            }
        }

        for (auto b : node->blocks()) {
            debug_tensors_for_block(b);
        }
    }
    //insert shape profile for block outputs
    for (size_t offset = 0; offset < block->return_node()->inputs().size(); offset++) {
        auto input_value = block->return_node()->input(offset);
        if (input_value->type()->isSubtypeOf(c10::TensorType::get())) {
            insert_debug_profile(block->return_node(), offset);
        }
    }
}

//DEPRECATED
std::vector<torch::jit::Node*> get_prim_if_user(torch::jit::Value* value) {
    std::vector<torch::jit::Node*> if_nodes;
    for (auto use : value->uses()) {
        if (is_dim_equal_if_node(use.user)) {
            if_nodes.emplace_back(use.user);
        }
    }
    //sort
    std::sort(if_nodes.begin(), if_nodes.end(), [&](torch::jit::Node* a, torch::jit::Node* b) {
        return a->isBefore(b);
    });
    return if_nodes;
}

//DEPRECATED
void IvalueAnalysis::prune_if_block(torch::jit::Block* block) {
    if (_evaluate_values_map.empty()) {
        return;
    }
    for (auto itr = block->nodes().begin(); itr != block->nodes().end(); itr++) {
        auto node = *itr;
        //itr++; // nonono, not here. the next node may be if node. and may be already destroyed below
        if (node->kind() == torch::jit::prim::profile && 
            node->outputs().size() == 0 && node->inputs().size() == 1) {
            auto evaluate_values = _evaluate_values_map.find(node->input(0));
            if (evaluate_values != _evaluate_values_map.end()) {
                auto bool_vector = evaluate_values->second;
                if (std::all_of(bool_vector.begin(), bool_vector.end(), 
                    [](bool i){ return i == true;})) {
                    //the result keep true during every data round
                    auto if_nodes = get_prim_if_user(node->input(0));
                    for (auto if_node: if_nodes) {
                        inline_if_body(if_node->blocks().at(0));
                    }
                }
                if (std::all_of(bool_vector.begin(), bool_vector.end(),
                    [](bool i){ return i == false;})) {
                    //the result keep false during every round
                    auto if_nodes = get_prim_if_user(node->input(1));
                    for (auto if_node: if_nodes) {
                        inline_if_body(if_node->blocks().at(1));
                    }
                }
            }
            //cause it has no output. so destroy it directly.
            node->destroy();
            _evaluate_values_map.erase(node->input(0));
        } else {
            for (torch::jit::Block* ib : node->blocks()) {
                prune_if_block(ib);
            }
        }
    }
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
