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
* @file coercion.h
* @author wangrui39@baidu.com
* @date Fri May 13 11:36:11 CST 2022
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

class CoercionConverter : public GpuConverter {
public:
    CoercionConverter() {}
    virtual ~CoercionConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::Int.float(float a) -> (int)", 
                "aten::Int.Tensor(Tensor a) -> (int)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::Int};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::Int.float(float a) -> (int)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::Int.Tensor(Tensor a) -> (int)", {1, 1}}});
        return result;
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
