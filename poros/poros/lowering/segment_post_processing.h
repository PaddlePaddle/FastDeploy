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
* @file segment_post_processing.h
* @author tianshaoqing@baidu.com
* @date 2022-05-27 11:11:18
* @brief
**/

#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief 有的子图输出的tensor原本是long类型，但是trtengine只支持int类型。
 *        那么就需要在engine后添加aten::to(long)的操作将其还原回去。
 *        避免有的op会强制检查long类型（例如：aten::index）
 *
 * @param [in] parent_graph : subgraph_node的owning_graph
 * @param [in] subgraph_node : 子图节点，类型必须是prim::CudaFusionGroup
 * @param [in] subgraph : 子图节点所对应的子图
 * 
 * @return 
 * @retval
**/
void subgraph_outputs_int2long(torch::jit::Graph* parent_graph,
                        torch::jit::Node& subgraph_node,
                        std::shared_ptr<torch::jit::Graph> subgraph);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu