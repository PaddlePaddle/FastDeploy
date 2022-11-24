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
* @file try_to_freeze_aten_size.cpp
* @author tianjinjin@baidu.com
* @date Fri Nov 26 11:35:16 CST 2021
* @brief
**/

#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

namespace {
using namespace torch::jit;

bool has_type_and_dim(const Value* value) {
    auto op = value->type()->cast<TensorType>();
    return op->sizes().size().has_value() && op->scalarType().has_value();
}

static std::string output_vec_size(const std::vector<int64_t>& vec_size) {
    if (vec_size.empty()) {
        return std::string("");
    } else {
        std::string output_str = "[";
        for (int64_t i : vec_size) {
            output_str += (std::to_string(i) + std::string(", "));
        }
        output_str.pop_back();
        output_str.pop_back();
        output_str.push_back(']');
        return output_str;
    }
}

struct FreezeAtenSize {
    FreezeAtenSize(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        GRAPH_DUMP("before freeze_aten_sizes Graph: ", graph_);
        bool changed = freeze_aten_sizes(graph_->block());
        if (changed) {
            ConstantPropagation(graph_);
            EliminateDeadCode(graph_);
            EliminateCommonSubexpression(graph_);
            ConstantPooling(graph_);
        }
        GRAPH_DUMP("after freeze_aten_sizes Graph: ", graph_);
    }

private:

    bool is_aten_size_node(Node* node) {
        if (node->kind() != aten::size) {
            return false;
        }
        //TODO: may be add more check situation
        return true;
    }

    void replace_int_list(Node* node, const std::vector<int64_t>& inplace_number) {
        LOG(INFO) << "try to replace the output of node :" << node_info(node)
                << " with constant value " << output_vec_size(inplace_number);
        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        auto int_list_const = graph_->insertConstant(inplace_number);
        node->outputs().at(0)->replaceAllUsesWith(int_list_const);
    }

    /**
     * try to calculate the result of aten::slice.
     * the schema of aten::slice is:
     * "aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]"
     * **/
    bool calculate_aten_slice(const torch::jit::Node* slice_node, 
                            const std::vector<int64_t>& input, 
                            std::vector<int64_t>& output) {

        if (slice_node->inputs().at(1)->node()->kind() != prim::Constant ||
            slice_node->inputs().at(2)->node()->kind() != prim::Constant ||
            slice_node->inputs().at(3)->node()->kind() != prim::Constant) {
            return false;
        }

        const int64_t input_len = input.size();
        auto maybe_start = toIValue(slice_node->inputs().at(1));
        auto start_index = maybe_start->isNone() ? 0 : maybe_start.value().toInt();
        const int64_t normalized_start = (start_index < 0) ? (input_len + start_index) : start_index;

        auto maybe_end = toIValue(slice_node->inputs().at(2));
        auto temp_end_index = maybe_end->isNone() ? INT64_MAX : maybe_end.value().toInt();
        auto end_idx = std::min(temp_end_index, input_len);
        const int64_t normalized_end = (end_idx < 0) ? (input_len + end_idx) : end_idx;

        if (normalized_end <= normalized_start) {
            return false;
        }
        int64_t step = toIValue(slice_node->inputs().at(3)).value().toInt();

        output.reserve(normalized_end - normalized_start);
        for (auto i = normalized_start; i < normalized_end;) {
            output.push_back(input[i]);
            i += step;
        }

        LOG(INFO) << "calculate_aten_slice done, input size: " << output_vec_size(input)
                << ", start_index: " << normalized_start
                << ", end_index: " << normalized_end
                << ", step: " << step
                << ", ouput size: " << output_vec_size(output);

        auto it = std::find_if(output.begin(), output.end(), [&](const int64_t& v) {return v == -1;});
        //不满足条件，output中有-1的值，说明存在dynamic的dim，不能替换成常量。
        if (it != output.end()) {
            return false;
        }

        return true;
    }

    /**
    * @brief 尝试解析aten::size的数据
    *        如果aten::size返回的list的后续使用，可以解除与动态变化的维度的关系，则相应值进行常量替换。
    **/ 
    bool try_to_freeze_aten_size(Node* node) {

        std::vector<int64_t> dims;
        if ((node->inputs()[0])->type()->isSubtypeOf(c10::TensorType::get()) &&
            has_type_and_dim(node->inputs()[0])) {
            gen_dims_for_tensor(node->inputs()[0], dims);
        } else {
            return false;
        }
        if (node->inputs().size() == 2) {
            return false;
        }

        //输入非tensor的场景，根据输入类型节点的类型简单判断。
        auto output_value = node->outputs()[0]; // should be a int[]
        auto users_count = (node->outputs()[0])->uses().size();

        //situation one: 如果aten::size算子本身的计算结果里面没有-1，则表示该tensor非dynamic，
        //则可以不管后面跟的是什么op，直接替换size的输出。
        auto it = std::find_if(dims.begin(), dims.end(), [&](const int64_t& v) {return v == -1;});
        if (it == dims.end()) {
            LOG(INFO) <<  "aten size output memebers are all constant situation, dim info: " << output_vec_size(dims);
            replace_int_list(node, dims);
            node->destroy();
            return true;
        }

        //situation two: aten::size is dynamic but the user is aten::slice
        if (users_count == 1 && (output_value->uses()[0]).user->kind() == aten::slice) {
            LOG(INFO) <<  "aten size user is aten::slice situation.";
            auto slice_node = (output_value->uses()[0]).user;
            std::vector<int64_t> sliced_list;
            if (calculate_aten_slice(slice_node, dims, sliced_list)) {
                //满足条件，替换节点
                replace_int_list(slice_node, sliced_list);
                //slice node 可以析构掉了
                slice_node->destroy();
                //当前的aten::size节点也可以析构掉了
                node->destroy();
                return true;
            }
        } else {
            LOG(INFO) << "not supported situation now.";
        }
        return false;
    }

    bool freeze_aten_sizes(Block* block) {
        bool changed = false;
        //fix bug: 可能连续后面几个节点被删除(比如aten::slice + aten::size), iterator改成从后往前迭代。
        for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
            // we might destroy the current node, so we need to pre-increment
            // the iterator
            Node* node = *it;
            ++it;
            for (Block* subblock : node->blocks()) {
                changed |= freeze_aten_sizes(subblock);
            }
            if (is_aten_size_node(node)) {
                LOG(INFO) << "find aten::size node: " << node_info(node);
                changed |= try_to_freeze_aten_size(node);
            }
        }
        return changed;
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void freeze_aten_size(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros freeze_aten_size passes";
    FreezeAtenSize fas(std::move(graph));
    fas.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu