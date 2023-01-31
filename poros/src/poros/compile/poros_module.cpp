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
* @file poros_module.cpp
* @author huangben@baidu.com
* @date 2021/08/05 11:39:03 CST 2021
* @brief 
**/

#include "poros/compile/poros_module.h"

namespace baidu {
namespace mirana {
namespace poros {

std::unique_ptr<PorosModule> Load(const std::string& filename, const PorosOptions& options) {
    torch::jit::Module module;
    try {
        module = torch::jit::load(filename);
    } catch (const c10::Error& e) {
        LOG(ERROR) << "error loading the model";
        return nullptr;
    }
    std::unique_ptr<PorosModule> poros_module(new PorosModule(module));
    poros_module->_options = options;

    if (options.device == GPU) {
        poros_module->to(at::kCUDA);
    }

    if (options.debug == true) {
        // when setting this, all the INFO level will be printed
        c10::ShowLogInfoToStderr();
    }
    
    return poros_module;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
