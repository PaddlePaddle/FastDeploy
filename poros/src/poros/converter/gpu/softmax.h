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
* @file softmax.h
* @author tianjinjin@baidu.com
* @date Tue Aug 24 17:15:33 CST 2021
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

class SoftmaxConverter : public GpuConverter {
public:
    SoftmaxConverter() {}
    virtual ~SoftmaxConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"};
    }
    
    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
     **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::softmax};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
