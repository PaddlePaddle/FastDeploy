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
* @file generate.h
* @author tianshaoqing@baidu.com
* @date Mon Dec 6 14:29:20 CST 2021
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

// Tensor zeros_like(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format);
class ZerosLikeConverter : public GpuConverter {
public:
    ZerosLikeConverter() {}
    virtual ~ZerosLikeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::zeros_like};
    }
};

// Tensor zeros(IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory); 
// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"
class ZerosConverter : public GpuConverter {
public:
    ZerosConverter() {}
    virtual ~ZerosConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::zeros};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", {1, 1}}});
    }
};

class OnesConverter : public GpuConverter {
public:
    OnesConverter() {}
    virtual ~OnesConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::ones};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", {1, 1}}});
    }
};

class FullConverter : public GpuConverter {
public:
    FullConverter() {}
    virtual ~FullConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::full};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", {1, 1}}});
    }
};

class ArangeConverter : public GpuConverter {
public:
    ArangeConverter() {}
    virtual ~ArangeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
                "aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::arange};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", {1, 1}}});
        return result;
    }
};

class TensorConverter : public GpuConverter {
public:
    TensorConverter() {}
    virtual ~TensorConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)",
                "aten::tensor.int(int t, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::tensor};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::tensor.int(int t, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)", {1, 1}}});
        return result;
    }
};

class LinspaceConverter : public GpuConverter {
public:
    LinspaceConverter() {}
    virtual ~LinspaceConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::linspace};
    }

    // aten::linspace schema changed in torch-1.11
    const std::vector<std::string> schema_string() {
        if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 11) {
            return {"aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",};
        } else {
            return {"aten::linspace(Scalar start, Scalar end, int steps, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",};
        }
    }
    
    // aten::linspace schema changed in torch-1.11
    bool assign_schema_attr() {
        if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 11) {
            return assign_schema_attr_helper({{"aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", {1, 1}}});
        } else {
            return assign_schema_attr_helper({{"aten::linspace(Scalar start, Scalar end, int steps, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", {1, 1}}});
        }
    }
};

class FulllikeConverter : public GpuConverter {
public:
    FulllikeConverter() {}
    virtual ~FulllikeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)",};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::full_like};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu