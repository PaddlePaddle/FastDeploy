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
* @file shape_handle.h
* @author tianjinjin@baidu.com
* @date Mon Nov 29 20:26:44 CST 2021
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

class AtenSizeConverter : public GpuConverter {
public:
    AtenSizeConverter() {}
    virtual ~AtenSizeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::size(Tensor self) -> (int[])",
                "aten::size.int(Tensor self, int dim) -> int"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::size};
    }
};

class ShapeastensorConverter : public GpuConverter {
public:
    ShapeastensorConverter() {}
    virtual ~ShapeastensorConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::_shape_as_tensor(Tensor self) -> (Tensor)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::_shape_as_tensor};
    }
};


class LenConverter : public GpuConverter {
public:
    LenConverter() {}
    virtual ~LenConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::len.Tensor(Tensor t) -> (int)",
                "aten::len.t(t[] a) -> (int)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::len};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::len.t(t[] a) -> (int)", {1, 1}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
