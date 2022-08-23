// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include "torch/script.h"
#include "torch/csrc/jit/jit_log.h"
#include "ATen/Context.h"

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
    // 该flag对tensorrt engine 有效, 默认为true
    // 当long_to_int=true，将相关value转成at::kInt进行处理（因为tensorrt不支持at::kLong 类型）
    // 该设置可能导致数据精度发生变化，如果效果不符合预期，请将该flag设置为false。
    bool long_to_int = true;
    //DynamicShapeOptions dynamic_shape_options;
    uint32_t max_workspace_size = 1ULL << 30;
    // XPU默认参数为-1，代表第一个可用设备
    int32_t device_id = -1;
    // 非const op个数阈值
    int32_t unconst_ops_thres = -1;
    // Nvidia TF32 computes inner products by rounding the inputs to 10-bit mantissas before multiplying, 
    // but accumulates the sum using 23-bit mantissas to accelerate the calculation.
    // note: It will work on ampere architecture (such as: A10), but may cause diff to the results.
    bool use_nvidia_tf32 = false;
    // preprocess mode
    // 0: use torch.jit.script
    // 1: use torhc.jit.trace
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

//compile api
std::unique_ptr<PorosModule> Compile(const torch::jit::Module& module, 
                                        const std::vector<std::vector<c10::IValue> >& prewarm_datas, 
                                        const PorosOptions& options);

//via porosmodule.save
std::unique_ptr<PorosModule> Load(const std::string& filename, const PorosOptions& options);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu