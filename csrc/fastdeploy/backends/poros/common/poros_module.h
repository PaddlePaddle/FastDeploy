/***************************************************************************
*
* Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
*
**************************************************************************/
/**
* @file poros_module.h
* @author huangben@baidu.com
* @date 2021/08/05  5 11:39:03 CST 2021
* @brief
**/

#pragma once

#include <string>
#include <torch/script.h>
#include <torch/csrc/jit/jit_log.h>
// #include <ATen/Context.h>

namespace baidu {
namespace mirana {
namespace poros {

enum Device : int8_t {
    GPU = 0,
    CPU,
    XPU,
    UNKNOW
};

struct PorosOptions {
    Device device = GPU;
    bool debug = false;
    bool use_fp16 = false;
    bool is_dynamic = false;
    bool long_to_int = true;
    uint64_t max_workspace_size = 1ULL << 30;
    int32_t device_id = -1;
    int32_t unconst_ops_thres = -1;
    bool use_nvidia_tf32 = false;
    // preprocess mode
    // 0: use torch.jit.script
    // 1: use torch.jit.trace
    int32_t preprocess_mode = 0;
};

class PorosModule : public torch::jit::Module {
public:
    PorosModule(torch::jit::Module module) : torch::jit::Module(module) {
    }
    ~PorosModule() = default;

    void to_device(Device device){
        _options.device = device;
    }

    //c10::IValue forward(std::vector<c10::IValue> inputs);
    //void save(const std::string& filename);
public:
    PorosOptions _options;

};

//via porosmodule.save
std::unique_ptr<PorosModule> Load(const std::string& filename, const PorosOptions& options);

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
