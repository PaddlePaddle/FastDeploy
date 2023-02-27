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
* @file expand.h
* @author tianjinjin@baidu.com
* @date Mon Aug 16 12:26:28 CST 2021
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

class ExpandConverter : public GpuConverter {
public:
    ExpandConverter() {}
    virtual ~ExpandConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
                "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)",};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::expand,
                torch::jit::aten::expand_as};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)", {1, 1}}});
    }
};

class RepeatConverter : public GpuConverter {
public:
    RepeatConverter() {}
    virtual ~RepeatConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::repeat(Tensor self, int[] repeats) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::repeat};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
