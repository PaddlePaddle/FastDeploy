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
* @file graph_prewarm.h
* @author tianjinjin@baidu.com
* @date Thu Mar 18 14:33:54 CST 2021
* @brief 
**/

#pragma once

//pytorch
#include <torch/script.h>

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief  首先，graph_prewarm 将给定的预热数据‘prewarm_datas’喂给添加了profile节点的graph。
 *              从而得到graph中每个节点的我们期望存储的信息(tensor类节点的dim信息/ bool类节点的值 等..)
 *         其次，graph_prewarm 基于这份存储的信息，进一步对graph进行图层面的优化，
 *         最后，graph_prewarm 返回一份完成了图优化的graph。
 * 
 *         first, graph_prewarm feed the given prewarm_datas to the graph which has added many profile nodes.
 *         we can get much information (dim information of tensor value / exact numerical value of 
 *         the bool value, etc...) about each node in the graph by profiling the graph。
 * 
 *         second,graph_prewarm optimize the given graph at the graph level based on the profile data we captured
 *         
 *         finally，graph_prewarm return the graph that has fully optimized in graph level  
 * 
 * @param [in] graph : the graph to be warmed
 * @param [in] prewarm_datas : prewarm data
 * @return prewarmed_graph
 **/
std::shared_ptr<torch::jit::Graph> graph_prewarm(
                                    std::shared_ptr<torch::jit::Graph>& graph, 
                                    const std::vector<std::vector<c10::IValue> >& prewarm_datas);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
