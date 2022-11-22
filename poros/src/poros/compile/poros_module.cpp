/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
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
