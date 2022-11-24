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
* @file eliminate_subgraph_useless_nodes.cpp
* @author tianshaoqing@baidu.com
* @date Thu May 16 19:49:02 CST 2022
* @brief
**/
#include "poros/lowering/lowering_pass.h"

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

using namespace torch::jit;

bool eliminate_subgraph_useless_nodes(std::shared_ptr<torch::jit::Graph> subgraph, 
                            torch::jit::Node& subgraph_node, 
                            const bool is_input) {
    AT_ASSERT(subgraph_node.kind() == torch::jit::prim::CudaFusionGroup);
    // init useless schema set
    std::unordered_set<c10::OperatorName> useless_schema_set;
    useless_schema_set.emplace(torch::jit::parseSchema(
        "aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False," 
        " bool copy=False, MemoryFormat? memory_format=None) -> Tensor").operator_name());
    useless_schema_set.emplace(torch::jit::parseSchema(
        "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None," 
        " bool non_blocking=False, bool copy=False) -> (Tensor(b|a))").operator_name());
    useless_schema_set.emplace(torch::jit::parseSchema("aten::contiguous(Tensor(a) self, *, "
    "MemoryFormat memory_format=contiguous_format) -> Tensor(a)").operator_name());
    useless_schema_set.emplace(torch::jit::parseSchema("aten::dropout(Tensor input, float p, "
    "bool train) -> Tensor").operator_name());
    useless_schema_set.emplace(torch::jit::parseSchema("aten::detach(Tensor(a) self) -> Tensor(a)").operator_name());
    useless_schema_set.emplace(torch::jit::parseSchema("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)")\
    .operator_name());
    
    // 由于execute_engine会对cpu的input做to(cuda)操作，所以子图输入前的to(cuda)可以删掉
    if (is_input) {
         // 先处理子图输入是aten::to.device的
        at::ArrayRef<torch::jit::Value*> node_inputs = subgraph_node.inputs();
        for (size_t i = 0; i < node_inputs.size(); i++) {
            torch::jit::Node* maybe_to_device_node = node_inputs[i]->node();
            // 如果在set中找到了aten::to.device，其type参数不是默认，则不能删
            if (maybe_to_device_node->kind() == torch::jit::aten::to &&
                useless_schema_set.count(maybe_to_device_node->schema().operator_name()) != 0 &&
                maybe_to_device_node->input(2)->node()->kind() == torch::jit::prim::Constant &&
                toIValue(maybe_to_device_node->inputs()[2])->isNone()) {

                auto to_device_users = maybe_to_device_node->output(0)->uses();
                // 需要保证aten::to.device output的所有user只有prim::CudaFusionGroup这一种
                bool all_users_cfg = true;
                for (size_t u = 0; u < to_device_users.size(); u++) {
                    if (to_device_users[u].user->kind() != prim::CudaFusionGroup) {
                        all_users_cfg = false;
                        break;
                    }
                }
                if (!all_users_cfg) {
                    continue;
                }
                // 给所有使用aten::to.device的子图替换输入
                for (size_t u = 0; u < to_device_users.size(); u++) {
                    to_device_users[u].user->replaceInput(to_device_users[u].offset, maybe_to_device_node->input(0));
                    LOG(INFO) << "Remove aten::to.device input[" << i << "] of subgraph: " << 
                    node_info(to_device_users[u].user) << ", which is useless.";
                }
                LOG(INFO) << "Destory node schema: [ " << maybe_to_device_node->schema() << " ]";
                // 删除aten::to.device
                maybe_to_device_node->destroy();
            }
        }
    } else {
        int unconst_nodes_num = 0;
        // 删除子图内部的aten::to.device
        auto cudafusion_subblock_nodes = subgraph->block()->nodes();
        for (auto c_it = cudafusion_subblock_nodes.begin(); c_it != cudafusion_subblock_nodes.end(); ) {
            torch::jit::Node* maybe_useless_node = *c_it;
            c_it++;
            if (maybe_useless_node->kind() != torch::jit::prim::Constant) {
                unconst_nodes_num++;
            }
            // 存在schema && 在useless_schema_set之中
            if (maybe_useless_node->maybeSchema() && 
                useless_schema_set.count(maybe_useless_node->schema().operator_name()) != 0) {
                bool is_useless_node = false;
                // 如果是aten::to.device，则需要额外判断scalartype是否为none，否则不能删
                if (maybe_useless_node->kind() == torch::jit::aten::to) {
                    if (maybe_useless_node->input(2)->node()->kind() == torch::jit::prim::Constant && 
                            toIValue(maybe_useless_node->inputs()[2])->isNone()) {
                        is_useless_node = true;
                    }
                // 对 rank=1 的 tensor 进行 aten::select 后接 aten::unsqueeze 的情况，
                // 原本 torch 中 rank=1 的 tensor select后 rank 会等于 0，
                // 有的模型（例如：faster-rcnn）会再加一次 unsqueeze 变回 rank=1，再进行其他操作。
                // 而 poros aten::select 的实现输出 nvtensor rank 依然是1，因此再 unsqueeze rank=2 就会出错。
                // 所以在子图里删掉这种情况的 aten::unsqueeze
                } else if (maybe_useless_node->kind() == torch::jit::aten::unsqueeze && 
                            maybe_useless_node->inputs().size() == 2 && 
                            maybe_useless_node->input(1)->node()->kind() == torch::jit::prim::Constant) {
                    int64_t unsqueeze_dim = toIValue(maybe_useless_node->input(1)).value().toInt();
                    torch::jit::Node* input0_node = maybe_useless_node->input(0)->node();
                    if (input0_node->kind() == torch::jit::aten::select && 
                        input0_node->outputs().size() == 1 && 
                        input0_node->output(0)->type()->isSubtypeOf(c10::TensorType::get()) && 
                        unsqueeze_dim == 0) {
                        auto select_output_type = input0_node->output(0)->type()->cast<c10::TensorType>();
                        // 通过c10::TensorType求rank
                        if (select_output_type->sizes().size().value() == 0) {
                            is_useless_node = true;
                        }
                    }
                } else {
                    // 其他节点暂时不用判断
                    is_useless_node = true;
                }

                if (is_useless_node) {
                    LOG(INFO) << "Remove " << node_info(maybe_useless_node) << " in subgraph: "<< 
                    node_info(&subgraph_node) << ", which is useless.";
                    LOG(INFO) << "Destory node schema: [ "<< maybe_useless_node->schema() << " ]";
                    maybe_useless_node->output(0)->replaceAllUsesWith(maybe_useless_node->input(0));
                    maybe_useless_node->destroy();
                    unconst_nodes_num--;
                }
            }
        }
        // 如果删完子图中的无用节点后只有constant节点，则返回false unmerge。
        if (unconst_nodes_num <= 0) {
            return false;
        }
    }
    return true;
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu