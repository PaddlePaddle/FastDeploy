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
* @file norm.h
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-23 20:33:45
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

class NormConverter : public GpuConverter {
public:
    NormConverter() {}

    virtual ~NormConverter() {}

    bool converter(TensorrtEngine *engine, const torch::jit::Node *node);

    //aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::norm};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor", {1, 1}}});
        return result;
    }

};

class FrobeniusNormConverter : public GpuConverter {
public:
    FrobeniusNormConverter() {}

    virtual ~FrobeniusNormConverter() {}

    bool converter(TensorrtEngine *engine, const torch::jit::Node *node);

    //aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::frobenius_norm};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor", {1, 1}}});
        return result;
    }

};

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
