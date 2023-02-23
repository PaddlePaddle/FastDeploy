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
* @file to.h
* @author wangrui39@baidu.com
* @date Saturday November 13 11:36:11 CST 2021
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

// Correspons to torch.tensor.to https://pytorch.org/docs/1.9.0/generated/torch.Tensor.to.html?highlight=#torch.to
class ToConverter : public GpuConverter {
public:
    ToConverter() {}
    virtual ~ToConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
                "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
                "aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
                "aten::to.dtype_layout(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
                "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))", 
        };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::to};
    }
};

class NumtotensorConverter : public GpuConverter {
public:
    NumtotensorConverter() {}
    virtual ~NumtotensorConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"prim::NumToTensor.Scalar(Scalar a) -> (Tensor)",
        };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::NumToTensor};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"prim::NumToTensor.Scalar(Scalar a) -> (Tensor)", {1, 1}}});
    }
};


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu