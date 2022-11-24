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
* @file select.h
* @author tianjinjin@baidu.com
* @date Tue Aug 24 16:31:28 CST 2021
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

class SelectConverter : public GpuConverter {
public:
    SelectConverter() {}
    virtual ~SelectConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)"};
    }

    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
     **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::select};
    }
};

class SliceConverter : public GpuConverter {
public:
    SliceConverter() {}
    virtual ~SliceConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
                "aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])"
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::slice};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])", {1, 1}}});
        return result;
    }
};

class EmbeddingConverter : public GpuConverter {
public:
    EmbeddingConverter() {}
    virtual ~EmbeddingConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::embedding};
    }
};

class NarrowConverter : public GpuConverter {
public:
    NarrowConverter() {}
    virtual ~NarrowConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
                "aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::narrow};
    }
};

class SplitConverter : public GpuConverter {
public:
    SplitConverter() {}
    virtual ~SplitConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]",
                "aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]",
                "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::split,
                torch::jit::aten::split_with_sizes,
                torch::jit::aten::unbind};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]", {0, 0}}});
        result &= assign_schema_attr_helper({{"aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]", {0, 0}}});
        result &= assign_schema_attr_helper({{"aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]", {0, 0}}});
        return result;
    }

};

class MaskedFillConverter : public GpuConverter {
public:
    MaskedFillConverter() {}
    virtual ~MaskedFillConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor",
                "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::masked_fill};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor", {1, 0}}});
    }
};

class GatherConverter : public GpuConverter {
public:
    GatherConverter() {}
    virtual ~GatherConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::gather};
    }
};

class IndexConverter : public GpuConverter {
public:
    IndexConverter() {}
    virtual ~IndexConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::index};
    }
};

class IndexPutConverter : public GpuConverter {
public:
    IndexPutConverter() {}
    virtual ~IndexPutConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::index_put};
    }
};

class ScatterConverter : public GpuConverter {
public:
    ScatterConverter() {}
    virtual ~ScatterConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::scatter};
    }
};

class ChunkConverter : public GpuConverter {
public:
    ChunkConverter() {}
    virtual ~ChunkConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"prim::ConstantChunk(...) -> (...)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::ConstantChunk};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"prim::ConstantChunk(...) -> (...)", {0, 0}}});
    }
};
}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
