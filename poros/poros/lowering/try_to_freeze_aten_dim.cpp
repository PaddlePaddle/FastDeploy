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
* @file: /poros/baidu/mirana/poros/src/poros/lowering/try_to_freeze_aten_dim.cpp
* @author: zhangfan51@baidu.com
* @data: 2022-03-24 16:02:50
* @brief: 
**/ 

#include "poros/lowering/lowering_pass.h"

#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

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

struct FreezeAtenDim {
    FreezeAtenDim(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

    void run() {
        try_to_freeze_aten_dim(graph_->block());
        // 运行一遍常量折叠
        torch::jit::ConstantPropagation(graph_);
        // torch::jit::ConstantPooling(graph_);
    }

private:
    void replace_aten_dim(Node* node, int inplace_number) {
        LOG(INFO) << "try to replace the output of node :" << node_info(node)
                << " with constant value " << inplace_number;
        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        auto len_const = graph_->insertConstant(inplace_number);
        node->outputs().at(0)->replaceAllUsesWith(len_const);
    }

    /*
    * @brief 尝试将aten::dim的返回值变成常量，如果 aten::dim的input的Tensor经过预热后维度确定，则可以使用常量代替
    **/    
    void try_to_freeze_aten_dim(Block* block) {
        auto it = block->nodes().begin();
        while (it != block->nodes().end()) {
            auto node = *it;
            ++it;  //先++it, node 可能destroy掉。
            for (auto block : node->blocks()) {
                try_to_freeze_aten_dim(block);
            }

            //只handle aten::dim 的场景
            if (node->kind() != aten::dim) {
                continue;
            }

            //输入是tensor，并且该tensor包含shape信息
            if ((node->inputs()[0])->type()->isSubtypeOf(c10::TensorType::get()) &&
                has_type_and_dim(node->inputs()[0])) {
                auto sizes = (node->inputs()[0])->type()->cast<TensorType>()->sizes();
                // if (sizes[0].has_value()) {
                    auto ndim = sizes.size();
                    replace_aten_dim(node, *ndim);
                // }
                continue;
            }
        }
    }

    std::shared_ptr<Graph> graph_;
};

} // namespace

void freeze_aten_dim(std::shared_ptr<torch::jit::Graph> graph) {
    LOG(INFO) << "Running poros freeze_aten_len passes";
    FreezeAtenDim pss(std::move(graph));
    pss.run();
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu