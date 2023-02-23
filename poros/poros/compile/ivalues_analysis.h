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
* @file ivalue_analysis.h
* @author tianjinjin@baidu.com
* @date Thu Mar 18 14:33:54 CST 2021
* @brief 
**/

#pragma once

#include <list>
#include <map>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>       //for Stack
#include <torch/csrc/jit/ir/ir.h>  //for ProfileOp
#include <torch/csrc/jit/runtime/profiling_record.h>  //for SetPartitioningHelper

#include "poros/context/poros_global.h"

namespace baidu {
namespace mirana {
namespace poros {

struct IvalueAnalysis {
    //disable copy and move op to avoid unexpected copy/move happened when in callback func
    IvalueAnalysis(const IvalueAnalysis&) = delete; 
    IvalueAnalysis(IvalueAnalysis&&) noexcept = delete;
    static std::unique_ptr<IvalueAnalysis> analysis_ivalue_for_graph(
        const std::shared_ptr<torch::jit::Graph>& graph);
        
    std::shared_ptr<torch::jit::Graph> profiled_graph_;
    std::mutex mutex_;
    size_t profiling_count_;
    // the key is a frame id
    // the value is a mapping from a Value in a graph to a profiled TensorType
    std::map<int64_t, std::map<torch::jit::Value*, std::vector<c10::TensorTypePtr>>> _profiled_types_per_frame;
    // meaning of key(int64_t) and value(Value*) are same to _profiled_types_per_frame.
    // vec<vec> records all of int(int[]) values against the Value* in a graph.
    std::map<int64_t, std::map<torch::jit::Value*, std::vector<std::vector<int64_t>>>> _int_intlist_values_per_frame;
    // std::map<int64_t, std::map<torch::jit::Value*, c10::IValue>>  _evaluate_values_per_frame;
    // we only store bool data. this may change in the future.
    std::map<torch::jit::Value*, std::vector<bool>> _evaluate_values_map;

    // 存储list类型的value相关信息
    ListSizeMap _list_size_map;
 
    std::shared_ptr<torch::jit::Graph> graph() const {
        return profiled_graph_;
    }

    // 拷贝dynamic信息到context
    void gen_value_dyanamic_shape();

    // 拷贝list信息到context
    void gen_list_size();

    // 拷贝int int[]值信息到context
    void gen_int_intlist_value();

 private:
    //ProfileIValueOp not supported  when in pytorch 1.7.x
    //so I have to rollback to ProfileOp, and have to copy the main function
    torch::jit::ProfileOp* create_profile_node(
                            const std::function<void(torch::jit::Stack&)>& fp, 
                            at::ArrayRef<torch::jit::Value*> inputs);

    void analysis_ivalue_for_block(torch::jit::Block* block);
    void insert_shape_profile(torch::jit::Node* node, size_t offset);
    void insert_eval_profile(torch::jit::Node* node, size_t offset);
    void insert_input_listsize_profile(torch::jit::Node* node, size_t offset);
    void insert_output_listsize_profile(torch::jit::Node* node, size_t offset);
    void insert_number_eval_profile(torch::jit::Node* node, size_t offset);

    /**
     * merge the tensortype list about one given value to a single merged tensortype
     * **/
    std::map<torch::jit::Value*, c10::TensorTypePtr> merge_tensor_type_per_frame(
                    std::map<torch::jit::Value*, std::vector<c10::TensorTypePtr>>& profiled_map);

    c10::SymbolicShape merge_symbolic_shapes(
                            const c10::SymbolicShape& new_sizes,
                            const c10::SymbolicShape& sym_shapes,
                            torch::jit::SetPartitioningHelper& partition_helper);

    //DEPRECATED
    //cut down some if block if the if-condition won't change during all the warm-up data
    void prune_if_block(torch::jit::Block* block);
    //DEPRECATED
    void debug_tensors_for_block(torch::jit::Block* block);
    //DEPRECATED
    void insert_debug_profile(torch::jit::Node* node, size_t offset);

    
    //be private.
    IvalueAnalysis(std::shared_ptr<torch::jit::Graph> g);
};


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
