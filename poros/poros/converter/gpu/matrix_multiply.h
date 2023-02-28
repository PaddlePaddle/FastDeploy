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
* @file matrix_multiply.h
* @author tianjinjin@baidu.com
* @date Wed Aug 18 20:30:19 CST 2021
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

class MatmulConverter : public GpuConverter {
public:
    MatmulConverter() {}
    virtual ~MatmulConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    nvinfer1::ITensor* converter(TensorrtEngine* engine,
                                const torch::jit::Node *node,
                                nvinfer1::ITensor* self,
                                nvinfer1::ITensor* other);

    const std::vector<std::string> schema_string() {
        return {"aten::matmul(Tensor self, Tensor other) -> Tensor"};
    }

    /**
     * TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::matmul};
    }
};

class BmmConverter : public GpuConverter {
public:
    BmmConverter() {}
    virtual ~BmmConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::bmm(Tensor self, Tensor mat2) -> Tensor"};
    }

    /**
     * TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::bmm};
    }
};

class AddmmConverter : public GpuConverter {
public:
    AddmmConverter() {}
    virtual ~AddmmConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"};
    }

    /**
     * TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
     * aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::addmm};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu