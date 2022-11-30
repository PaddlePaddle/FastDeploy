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
* @file engine_context.h
* @author tianjinjin@baidu.com
* @date Fri Jul 23 11:21:10 CST 2021
* @brief 
**/

#pragma once

#include <mutex>
#include <unordered_map>
#include "torch/script.h"

namespace baidu {
namespace mirana {
namespace poros {

template <class T>
class EngineContext {
public:
    explicit EngineContext() {}

    T* get_tensor(const torch::jit::Value* value) {
        auto it = _value_tensor_map.find(value);
        if (it == _value_tensor_map.end()) {
            return nullptr;
        }
        return it->second;
    }

    bool set_tensor(const torch::jit::Value* value,  T* tensor) {
        if (value != nullptr && tensor != nullptr) {
            _value_tensor_map[value] = tensor;
            return true;
        }
        return false;
    }

    bool get_tensorlist(const torch::jit::Value* value, std::vector<T*>& tensorlist) {
        auto it = _value_tensorlist_map.find(value);
        if (it == _value_tensorlist_map.end()) {
            return false;
        }
        tensorlist = it->second;
        return true;
    }

    bool set_tensorlist(const torch::jit::Value* value,  std::vector<T*> tensorlist) {
        if (value != nullptr) {
            _value_tensorlist_map[value] = tensorlist;
            return true;
        }
        return false;
    }

    torch::jit::IValue get_constant(const torch::jit::Value* value) {
        auto it = _value_constant_map.find(value);
        if (it != _value_constant_map.end()) {
            return it->second;
        } else {
            return torch::jit::IValue();
        }
    }
    
    bool set_constant(const torch::jit::Value* value,  torch::jit::IValue constant) {
        if (value != nullptr) {
            _value_constant_map[value] = constant;
            return true;
        }
        return false;
    }

private:
    std::string _engine_id;
    //value <-> nvtensor
    std::unordered_map<const torch::jit::Value*, T*> _value_tensor_map;
    //value <-> nvtensor list
    //std::unordered_map<const torch::jit::Value*, c10::List<T*>> _value_tensorlist_map;
    std::unordered_map<const torch::jit::Value*, std::vector<T*>> _value_tensorlist_map;
    //value <-> others
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue> _value_constant_map;
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
