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
* @file non_converterable.h
* @author tianjinjin@baidu.com
* @date Thu Aug 26 10:24:14 CST 2021
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

class ContiguousConverter : public GpuConverter {
public:
    ContiguousConverter() {}
    virtual ~ContiguousConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::contiguous};
    }
};

class DropoutConverter : public GpuConverter {
public:
    DropoutConverter() {}
    virtual ~DropoutConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::dropout(Tensor input, float p, bool train) -> Tensor",
                "aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
                "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
                "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
     * "aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
     * 
     * some stange err msg like : feature_alpha_dropout_ is not a member of 'torch::jit::aten' 
     * 
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::dropout,
                torch::jit::aten::dropout_,
                torch::jit::aten::feature_dropout,
                torch::jit::aten::feature_alpha_dropout,};
    }

};

// aten::IntImplicit(Tensor a) -> (int)
class IntimplicitConverter : public GpuConverter {
public:
    IntimplicitConverter() {}
    virtual ~IntimplicitConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::IntImplicit(Tensor a) -> (int)"};
    }
    
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::IntImplicit};
    }

};

// prim::tolist
class TolistConverter : public GpuConverter {
public:
    TolistConverter() {}
    virtual ~TolistConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::tolist kind node has no schema
    const std::vector<std::string> schema_string() {
        return {};
    }
    
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::tolist};
    }

};

// aten::detach
class DetachConverter : public GpuConverter {
public:
    DetachConverter() {}
    virtual ~DetachConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::tolist kind node has no schema
    const std::vector<std::string> schema_string() {
        return {"aten::detach(Tensor(a) self) -> Tensor(a)"};
    }
    
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::detach};
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
