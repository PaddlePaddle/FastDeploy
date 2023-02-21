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
* @file clone.h
* @author tianshaoqing@baidu.com
* @date Tue Nov 23 12:26:28 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include "torch/script.h"

#include "poros/converter/gpu/gpu_converter.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {

class CloneConverter : public GpuConverter {
public:
    CloneConverter() {}
    virtual ~CloneConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    const std::vector<std::string> schema_string() {
        return {"aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::clone};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
