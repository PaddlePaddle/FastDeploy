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
* @file add.h
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

class AddConverter : public GpuConverter {
public:
    AddConverter() {}
    virtual ~AddConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    const std::vector<std::string> schema_string() {
        return {"aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
                "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
                "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
                "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
                "aten::add.int(int a, int b) -> (int)",
                "aten::add.t(t[] a, t[] b) -> (t[])"
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::add,
                torch::jit::aten::add_};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::add.int(int a, int b) -> (int)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::add.t(t[] a, t[] b) -> (t[])", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)", {1, 1}}}); 
        return result;
    }
};

class SubConverter : public GpuConverter {
public:
    SubConverter() {}
    virtual ~SubConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
                "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
                "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
                "aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
                "aten::sub.int(int a, int b) -> (int)",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::sub,
                torch::jit::aten::sub_};
    }
    
    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::sub.int(int a, int b) -> (int)", {1, 1}}});
    }
};

class RsubConverter : public GpuConverter {
public:
    RsubConverter() {}
    virtual ~RsubConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)",
                "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::rsub};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
