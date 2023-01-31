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
* @file logical.h
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-17 18:32:23
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

class AndConverter : public GpuConverter {
public:
    AndConverter() {}
    virtual ~AndConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::__and__,
                torch::jit::aten::__iand__,
                torch::jit::aten::bitwise_and,
                };
    }
};

class OrConverter : public GpuConverter {
public:
    OrConverter() {}
    virtual ~OrConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {
                "aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return  {torch::jit::aten::__or__,
                 torch::jit::aten::__ior__,
                 torch::jit::aten::bitwise_or,
        };
    }
};

class XorConverter : public GpuConverter {
public:
    XorConverter() {}
    virtual ~XorConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {
                "aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::__xor__,
                torch::jit::aten::__ixor__,
                torch::jit::aten::bitwise_xor,
        };
    }
};

class NotConverter : public GpuConverter {
public:
    NotConverter() {}
    virtual ~NotConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::bitwise_not(Tensor self) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::bitwise_not(Tensor self) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {
                torch::jit::aten::bitwise_not,
        };

    }
};

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
