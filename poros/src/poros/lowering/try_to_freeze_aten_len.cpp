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
* @file try_to_freeze_aten_len.cpp
* @author tianjinjin@baidu.com
* @date Sun Sep 26 20:00:01 CST 2021
* @brief
**/

#include "poros/lowering/lowering_pass.h"

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

struct FreezeAtenLen {
    FreezeAtenLen(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        try_to_freeze_aten_len(graph_->block());
    }

private:
    void replace_aten_len(Node* node, int inplace_number) {
        LOG(INFO) << "try to replace the output of node :" << node_info(node)
                << " with constant value " << inplace_number;
        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        auto len_const = graph_->insertConstant(inplace_number);
        node->outputs().at(0)->replaceAllUsesWith(len_const);
    }

    bool try_to_replace_listconstruct_len(Node* node, Node* len_node) {
        if (node->kind() != prim::ListConstruct) {
            return false;
        }
        for (auto &use : node->outputs()[0]->uses()) {
            if (use.user->owningBlock() != node->owningBlock() ||
                use.user->kind() == aten::append) {
                return false;
            }
        }
        replace_aten_len(len_node, node->inputs().size());
        return true;
    }

    /**
    * @brief 尝试将aten::len的返回值变成常量，当前支持的场景：
    *       1. aten::len 的输入是一个tensor(前提是我们认为tensor的size可能是dynamic的，但是len是确定的)
    *       2. aten::len 的输入是prim::ListConstruct构建的list，当list的长度可以明确的时候，进行常量替换。
    *       3. aten::len 的输入是aten::unbind，根据该算子的语义，获取其len并替换。
    *       4. aten::len 的输入是aten::meshgrid，由于这类算子不改变输入的len信息，进一步获取算子的输入，尝试进行常量替换。
    **/    
    void try_to_freeze_aten_len(Block* block) {
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //先++it, node 可能destroy掉。
            for (auto block : node->blocks()) {
                try_to_freeze_aten_len(block);
            }

            //只handle aten::len 的场景
            if (node->kind() != aten::len) {
                continue;
            }
            
            //输入是一个tensor的场景, aten::len的结果
            if ((node->inputs()[0])->type()->isSubtypeOf(c10::TensorType::get()) &&
                has_type_and_dim(node->inputs()[0])) {
                LOG(INFO) << "input is tensor situation.";
                auto sizes = (node->inputs()[0])->type()->cast<TensorType>()->sizes();
                if (sizes[0].has_value()) {
                    int len = sizes[0].value();
                    replace_aten_len(node, len);
                }
                continue;
                // std::vector<int64_t> dims;
                // if (gen_dims_for_tensor(node->inputs()[0], dims)) {
                //     int len = (dims.size()) & INT_MAX;
                //     replace_aten_len(node, len);
                // }
                // continue;
            }

            //输入非tensor的场景，根据输入类型节点的类型简单判断。
            auto input_node = (node->inputs()[0])->node();
            switch (input_node->kind()) {
            // unbind: 等于第一个输入的
                case aten::unbind: {
                    LOG(INFO) <<  "input is produced by aten::unbind situation.";
                    if (has_type_and_dim(input_node->inputs()[0])) {
                        std::vector<int64_t> dims;
                        if (gen_dims_for_tensor(input_node->inputs()[0], dims) &&
                            input_node->inputs()[1]->node()->kind() == prim::Constant) {
                            int dim = toIValue(input_node->inputs()[1]->node()->output()).value().toInt();
                            dim = dim < 0 ? dims.size() + dim : dim;
                            //非dynamic的维度
                            if (dims[dim] != -1) {
                                auto len = dims[dim];
                                replace_aten_len(node, len);
                                torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
                                auto len_const = graph_->insertConstant(len);
                                node->outputs().at(0)->replaceAllUsesWith(len_const);
                            }
                        }
                    }
                    break;
                }
                //这些op的输入输出的len不会发生变化。再找一下这类op的输入。
                case aten::meshgrid: {
                    LOG(INFO) <<  "input is produced by aten:meshgrid situation.";
                    if ((input_node->inputs()[0])->node()->kind() == prim::ListConstruct) {
                        try_to_replace_listconstruct_len((input_node->inputs()[0])->node(), node);
                    }
                    break;
                }
                //prim::ListConstruct 的情况
                case prim::ListConstruct: {
                    LOG(INFO) <<  "input is produced by prim::ListConstruct situation.";
                    try_to_replace_listconstruct_len(input_node, node);
                    break;
                }
                default: {
                    //遇到目前不支持的类型，直接返回，不做处理。
                    LOG(INFO) <<  "unsupported situation. input_node is: " << node_info(input_node);
                    break;
                }         
            }
        }
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void freeze_aten_len(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros freeze_aten_len passes";
    FreezeAtenLen fal(std::move(graph));
    fal.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu