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
* @file stack.h
* @author tianjinjin@baidu.com
* @date Tue Sep  7 15:09:14 CST 2021
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

class StackConverter : public GpuConverter {
public:
    StackConverter() {}
    virtual ~StackConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
                "aten::vstack(Tensor[] tensors) -> Tensor"
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::stack,
                torch::jit::aten::vstack,
                };
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", {1, 1}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
