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
* @file engine.cpp
* @author huangben@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/engine/iengine.h"
#include "poros/context/poros_global.h"
#include "poros/converter/iconverter.h"

namespace baidu {
namespace mirana {
namespace poros {

bool IEngine::is_node_supported(const torch::jit::Node* node) {
    auto converter_map = PorosGlobalContext::instance().get_converter_map(who_am_i());
    if (converter_map != nullptr && converter_map->node_converterable(node)) {
        return true;
    } else {
        if (node->kind() != torch::jit::prim::Loop && 
                node->kind() != torch::jit::prim::If &&
                node->kind() != torch::jit::prim::CudaFusionGroup &&
                node->kind() != torch::jit::prim::Param) {
            LOG(INFO) << "not supported node: " << node->kind().toQualString()
                        << ", detail info: " << *node;
        }
        // auto convertableItr = get_non_convertable_nodes().find(node->kind().toQualString());
        // if (convertableItr != get_non_convertable_nodes().end()) {
        //     return true;
        // }
        return false;
    }
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
