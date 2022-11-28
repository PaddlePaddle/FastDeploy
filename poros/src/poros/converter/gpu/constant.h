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
* @file constant.h
* @author tianjinjin@baidu.com
* @date Tue Jul 27 11:24:21 CST 2021
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

class ConstantConverter : public GpuConverter {
public:
    ConstantConverter() {}
    virtual ~ConstantConverter() {}

    /**
     * @brief converter, 将node转换成layer，添加到engine的组网逻辑里
     * @retval 0 => success, <0 => fail
     **/ 
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::Constant kind node has no schema
    const std::vector<std::string> schema_string() {
        return {};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::Constant};
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
