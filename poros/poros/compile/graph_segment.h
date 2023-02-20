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
* @file graph_segment.h
* @author tianjinjin@baidu.com
* @date Thu Mar 18 14:33:54 CST 2021
* @brief 
**/

#pragma once

//pytorch
#include <torch/script.h>

#include "poros/engine/iengine.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief  graph_segment fullfil the segmentation func of given graph
 * @param [in/out] graph : the graph to be segmented
 * @param [in] engine : backend engine
 * @return
 **/
void graph_segment(std::shared_ptr<torch::jit::Graph>& graph, IEngine* engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
