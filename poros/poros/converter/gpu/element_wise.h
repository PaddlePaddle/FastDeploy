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
* @file element_wise.h
* @author tianjinjin@baidu.com
* @date Fri Aug 27 15:32:36 CST 2021
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

class GreaterOrLessConverter : public GpuConverter {
public:
    GreaterOrLessConverter() {}
    virtual ~GreaterOrLessConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);
    
    // note: gt.int, lt.int, ge.int le.int maybe should not support better.
    // because they usually appear with if or loop.
    const std::vector<std::string> schema_string() {
        return {"aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::gt.int(int a, int b) -> (bool)",
                "aten::lt.int(int a, int b) -> (bool)",
                "aten::ge.int(int a, int b) -> (bool)",
                "aten::le.int(int a, int b) -> (bool)",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::gt,
                torch::jit::aten::lt,
                torch::jit::aten::ge,
                torch::jit::aten::le};
    }

private:
    nvinfer1::ITensor* scalar_to_nvtensor(TensorrtEngine* engine, at::Scalar s);
};

class EqualOrNotequalConverter : public GpuConverter {
public:
    EqualOrNotequalConverter() {}
    virtual ~EqualOrNotequalConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    // note: eq.int, ne.int maybe should not support better.
    // because they usually appear with if or loop.
    const std::vector<std::string> schema_string() {
        return {"aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
                "aten::eq.int(int a, int b) -> (bool)",
                "aten::ne.int(int a, int b) -> (bool)"
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::eq,
                torch::jit::aten::ne,
                };
    }
};

class PowOrFloordivideConverter : public GpuConverter {
public:
    PowOrFloordivideConverter() {}
    virtual ~PowOrFloordivideConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor",
                "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
                //"aten::floor_divide(Tensor self, Tensor other) -> Tensor",
                //"aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor",
     * 
     * aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::pow,
                //torch::jit::aten::floor_divide,
                };
    }
};

class ClampConverter : public GpuConverter {
public:
    ClampConverter() {}
    virtual ~ClampConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
                "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
                "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor",
     * "aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor",
     * "aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor",
     * 
     * "aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
     * 
     * "aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)",
     * "aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)"
     * "aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::clamp,
                torch::jit::aten::clamp_min,
                torch::jit::aten::clamp_max,
                };
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
