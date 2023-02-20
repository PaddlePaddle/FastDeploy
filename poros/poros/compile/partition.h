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
* @file partition.h
* @author tianjinjin@baidu.com
* @date Thu Jun  3 14:57:58 CST 2021
* @brief 
**/

#pragma once

#include "torch/script.h"

#include "poros/engine/iengine.h"

namespace baidu {
namespace mirana {
namespace poros {

bool is_node_fusable(const torch::jit::Node* node, IEngine* engine);
bool is_node_fusable(const torch::jit::Node* fusion, 
                    const torch::jit::Node* node, 
                    IEngine* engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
