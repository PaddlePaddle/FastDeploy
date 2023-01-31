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
* @file activation.h
* @author tianjinjin@baidu.com
* @date Wed Jul 28 15:24:51 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include <torch/script.h>
#include <torch/version.h>

#include "poros/converter/gpu/gpu_converter.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {

class ActivationConverter : public GpuConverter {
public:
    ActivationConverter() {}
    virtual ~ActivationConverter() {}
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::relu(Tensor self) -> Tensor",
        "aten::relu_(Tensor(a!) self) -> Tensor(a!)",
        "aten::relu6(Tensor self) -> (Tensor)",
        "aten::relu6_(Tensor(a!) self) -> Tensor(a!)",
        "aten::sigmoid(Tensor self) -> Tensor",
        "aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)",
        "aten::tanh(Tensor self) -> Tensor",
        "aten::tanh_(Tensor(a!) self) -> Tensor(a!)",
        "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor",
        "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
        "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)",
        "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor",
        "aten::silu(Tensor self) -> Tensor"};
    }

    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * //said 'leaky_relu_' is not a member of 'torch::jit::aten' and i don't know why
     * "aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)", 
     * */
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::relu,
                torch::jit::aten::relu_,
                torch::jit::aten::relu6,
                torch::jit::aten::relu6_,
                torch::jit::aten::sigmoid,
                torch::jit::aten::sigmoid_,
                torch::jit::aten::tanh,
                torch::jit::aten::tanh_,
                torch::jit::aten::leaky_relu,
                torch::jit::aten::hardtanh,
                torch::jit::aten::hardtanh_,
                torch::jit::aten::elu,
                torch::jit::aten::silu};
    }
};

class GeluActivationConverter : public GpuConverter {
public:
    GeluActivationConverter() {}
    virtual ~GeluActivationConverter() {}
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);
    const std::vector<std::string> schema_string() {
        // aten::gelu schema changed in torch-1.12
        if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 12) {
            return {"aten::gelu(Tensor self) -> Tensor"};
        } else {
            return {"aten::gelu(Tensor self, *, str approximate='none') -> Tensor"};
        }
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::gelu};
    }
};

class PreluActivationConverter : public GpuConverter {
public:
    PreluActivationConverter() {}
    virtual ~PreluActivationConverter() {}
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);
    const std::vector<std::string> schema_string() {
        return {"aten::prelu(Tensor self, Tensor weight) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::prelu};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
