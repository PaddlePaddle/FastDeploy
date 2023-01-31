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
* @file unary.h
* @author tianjinjin@baidu.com
* @date Mon Sep  6 20:23:14 CST 2021
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

class UnaryConverter : public GpuConverter {
public:
    UnaryConverter() {}
    virtual ~UnaryConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::cos(Tensor self) -> Tensor",
                "aten::acos(Tensor self) -> Tensor",
                "aten::cosh(Tensor self) -> Tensor",
                "aten::sin(Tensor self) -> Tensor",
                "aten::asin(Tensor self) -> Tensor",
                "aten::sinh(Tensor self) -> Tensor",
                "aten::tan(Tensor self) -> Tensor",
                "aten::atan(Tensor self) -> Tensor",
                "aten::abs(Tensor self) -> Tensor",
                "aten::floor(Tensor self) -> Tensor",
                "aten::reciprocal(Tensor self) -> Tensor",
                "aten::log(Tensor self) -> Tensor",
                "aten::ceil(Tensor self) -> Tensor",
                "aten::sqrt(Tensor self) -> Tensor",
                "aten::exp(Tensor self) -> Tensor",
                "aten::neg(Tensor self) -> Tensor",
                "aten::erf(Tensor self) -> Tensor",
                "aten::asinh(Tensor self) -> Tensor",
                "aten::acosh(Tensor self) -> Tensor",
                "aten::atanh(Tensor self) -> Tensor",
                "aten::log2(Tensor self) -> (Tensor)",
                "aten::log10(Tensor self) -> (Tensor)",
                "aten::floor.float(float a) -> (int)",
                "aten::round(Tensor self) -> (Tensor)"
            };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
     * "ALL OF THEIR .out CONVERTERS IS NOT SUPPORTED"
     * "ALL OF THEIR .out CONVERTERS IS NOT SUPPORTED"
     * "ALL OF THEIR .out CONVERTERS IS NOT SUPPORTED"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::cos,
                torch::jit::aten::acos,
                torch::jit::aten::cosh,
                torch::jit::aten::sin,
                torch::jit::aten::asin,
                torch::jit::aten::sinh,
                torch::jit::aten::tan,
                torch::jit::aten::atan,
                torch::jit::aten::abs,
                torch::jit::aten::floor,
                torch::jit::aten::reciprocal,
                torch::jit::aten::log,
                torch::jit::aten::ceil,
                torch::jit::aten::sqrt,
                torch::jit::aten::exp,
                torch::jit::aten::neg,
                torch::jit::aten::erf,
                torch::jit::aten::asinh,
                torch::jit::aten::acosh,
                torch::jit::aten::atanh,
                torch::jit::aten::log2,
                torch::jit::aten::log10,
                torch::jit::aten::round
                };
    }
    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::floor.float(float a) -> (int)", {1, 1}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
