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
* @file mul_div.h
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

class MulConverter : public GpuConverter {
public:
    MulConverter() {}
    virtual ~MulConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
                "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
                "aten::mul.int(int a, int b) -> (int)",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::mul,
                torch::jit::aten::mul_};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::mul.int(int a, int b) -> (int)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::mul.Scalar(Tensor self, Scalar other) -> Tensor", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", {1, 1}}});
        return result;
    }
};

class DivConverter : public GpuConverter {
public:
    DivConverter() {}
    virtual ~DivConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)",
                "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
                "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
                "aten::div.int(int a, int b) -> (float)",
                "aten::div(Scalar a, Scalar b) -> (float)"
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor",
     * "aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"
     * "aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::div,
                torch::jit::aten::div_};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::div.int(int a, int b) -> (float)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::div(Scalar a, Scalar b) -> (float)", {1, 1}}});
        return result;
    }

};

class FloordivConverter : public GpuConverter {
public:
    FloordivConverter() {}
    virtual ~FloordivConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::floordiv.int(int a, int b) -> (int)",
                "aten::__round_to_zero_floordiv.int(int a, int b) -> (int)"
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::floordiv,
                torch::jit::aten::__round_to_zero_floordiv};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::floordiv.int(int a, int b) -> (int)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::__round_to_zero_floordiv.int(int a, int b) -> (int)", {1, 1}}});
        return result;
    }
};


class RemainderConverter : public GpuConverter {
public:
    RemainderConverter() {}
    virtual ~RemainderConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::remainder.Scalar(Tensor self, Scalar other) -> (Tensor)",
                "aten::remainder.Tensor(Tensor self, Tensor other) -> (Tensor)",
        };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::remainder};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::remainder.Scalar(Tensor self, Scalar other) -> (Tensor)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::remainder.Tensor(Tensor self, Tensor other) -> (Tensor)", {1, 1}}});
        return result;
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
