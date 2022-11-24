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
* @file gpu_converter.cpp
* @author tianshaoqing@baidu.com
* @date Mon Dec 27 11:24:21 CST 2021
* @brief 
**/

#include "poros/converter/gpu/gpu_converter.h"

#include "poros/converter/gpu/weight.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {
    
bool GpuConverter::check_inputs_tensor_scalar(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    bool has_tensor_scalar = false;
    // 检查int或int[]类型的输入是否包含nvtensor
    for (size_t i = 0; i < inputs.size(); i++) {
        const torch::jit::Value* node_input = inputs[i];
        // int, int[] or float32
        if (node_input->type()->kind() == c10::TypeKind::IntType ||
            node_input->type()->isSubtypeOf(c10::ListType::ofInts()) ||
            node_input->type()->str() == "float") {
            if (engine->context().get_tensor(node_input) != nullptr) {
                // 检查成功立刻停止循环
                LOG(INFO) << node_info(node) << ": inputs[" << i << "] is tensor scalar.";
                has_tensor_scalar = true;
                break;
            }
        }
    }
    // 如果int或int[]中包含nvtensor, 将所有输入int与nvtensor建立映射到map中
    if (has_tensor_scalar) {
        _tensor_scalar_map.clear();
        for (size_t i = 0; i < inputs.size(); i++) {
            const torch::jit::Value* node_input = inputs[i];
            // int or int[] 
            // 2022.5.13 @wangrui39: 这里加了float，这个情况会用在aten::div(Scalar a, Scalar b) -> (float)
            if (node_input->type()->kind() == c10::TypeKind::IntType ||
                node_input->type()->isSubtypeOf(c10::ListType::ofInts()) ||
                node_input->type()->str() == "float") {
                nvinfer1::ITensor* temp = engine->context().get_tensor(inputs[i]);
                // 若直接获取到了nvtensor, 直接建立映射
                if (temp != nullptr) {
                    _tensor_scalar_map.emplace(inputs[i], temp);
                } else {
                    // 若未获取int或int[]对应到nvtensor, get其ivalue值再转成nvtensor, 建立映射关系
                    torch::jit::IValue temp_ivalue = engine->context().get_constant(inputs[i]);
                    if (temp_ivalue.isInt()) {
                        int64_t temp_int = temp_ivalue.toScalar().to<int64_t>();
                        _tensor_scalar_map.emplace(inputs[i], 
                        tensor_to_const(engine, torch::tensor({temp_int}, torch::kInt)));
                    } else if (temp_ivalue.type()->str() == "float") {
                        float temp_float = temp_ivalue.toScalar().to<float>();
                        _tensor_scalar_map.emplace(inputs[i], 
                        tensor_to_const(engine, torch::tensor({temp_float}, torch::kFloat)));
                    } else if (temp_ivalue.isIntList()){
                        _tensor_scalar_map.emplace(inputs[i], 
                        tensor_to_const(engine, torch::tensor(temp_ivalue.toIntList().vec(), torch::kInt)));
                    } else {
                        // 若获取ivalue也失败, 则建立int与空指针的关系, 外部获取后需判断
                        _tensor_scalar_map.emplace(inputs[i], nullptr);
                        LOG(FATAL) << node_info(node) + std::string(" input[") + 
                                        std::to_string(i) + std::string("] get int ivalue false.");
                    }
                }
            }
        }
    }
    return has_tensor_scalar;
}

nvinfer1::ITensor* GpuConverter::get_tensor_scalar(const torch::jit::Value* value) {
    auto it = _tensor_scalar_map.find(value);
    if (it == _tensor_scalar_map.end()) {
        return nullptr;
    }
    return it->second;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
