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
* @file segment_post_processing.cpp
* @author tianshaoqing@baidu.com
* @date Thu May 27 11:13:02 CST 2022
* @brief
**/

#include "poros/lowering/segment_post_processing.h"

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

using namespace torch::jit;

void subgraph_outputs_int2long(torch::jit::Graph* parent_graph,
                        torch::jit::Node& subgraph_node,
                        std::shared_ptr<torch::jit::Graph> subgraph) {
    AT_ASSERT(subgraph_node.kind() == torch::jit::prim::CudaFusionGroup);
    // 检查子图的每个Tensor的输出类型
    for (size_t i = 0; i < subgraph->outputs().size(); i++) {
        torch::jit::Value* output_value = subgraph->outputs()[i];
        if (output_value->type()->isSubtypeOf(c10::TensorType::get())) {
            auto subgraph_output_type = output_value->type()->cast<c10::TensorType>();
            if (subgraph_output_type->scalarType() == at::ScalarType::Long) {
                // 如果子图Tensor输出是Long，则添加aten::to.dtype，schema如下：
                // aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
                LOG(INFO) << "Find output type is Long, which is " << node_info(&subgraph_node) << " output[" << i <<
                "] %" << output_value->debugName() << ". Add aten::to(Long) node.";
                torch::jit::Node* to_long_node = parent_graph->create(torch::jit::aten::to, 1);
                to_long_node->insertAfter(&subgraph_node);
                to_long_node->addInput(subgraph_node.output(i));
                // 不用setInsertPoint的话默认将constant插入到图的末尾，movebefore将constant移到to_long_node之前
                torch::jit::Value* false_value = parent_graph->insertConstant(false);
                false_value->node()->moveBefore(to_long_node);
                torch::jit::Value* type_value = parent_graph->insertConstant(c10::ScalarType::Long);
                type_value->node()->moveBefore(to_long_node);
                torch::jit::Node* none_node = parent_graph->createNone();
                none_node->insertBefore(to_long_node);

                to_long_node->addInput(type_value);
                to_long_node->addInput(false_value);
                to_long_node->addInput(false_value);
                to_long_node->addInput(none_node->output(0));

                // must set output type
                to_long_node->output(0)->setType(subgraph_output_type);
                subgraph_node.output(i)->replaceAllUsesAfterNodeWith(to_long_node, to_long_node->output(0));
            }
        }
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu