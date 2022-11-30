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
* @file shuffle.h
* @author tianjinjin@baidu.com
* @date Wed Aug 18 15:37:48 CST 2021
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

class FlattenConverter : public GpuConverter {
public:
    FlattenConverter() {}
    virtual ~FlattenConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)"};
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
     * aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
     * aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
     * **/

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::flatten};
    }
};


class PermuteViewConverter : public GpuConverter {
public:
    PermuteViewConverter() {}
    virtual ~PermuteViewConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
                "aten::view(Tensor(a) self, int[] size) -> Tensor(a)"};
    }

    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
     **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::permute,
                torch::jit::aten::view};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::view(Tensor(a) self, int[] size) -> Tensor(a)", {1, 1}}});
    }
};

class ReshapeConverter : public GpuConverter {
public:
    ReshapeConverter() {}
    virtual ~ReshapeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::reshape};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)", {1, 1}}});
    }
};

class TransposeConverter : public GpuConverter {
public:
    TransposeConverter() {}
    virtual ~TransposeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"};
    }

    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
     * aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
     **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::transpose};
    }
};

class AtenTConverter : public GpuConverter {
public:
    AtenTConverter() {}
    virtual ~AtenTConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::t(Tensor(a) self) -> Tensor(a)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::t};
    }
};

class PixelShuffleConverter : public GpuConverter {
public:
    PixelShuffleConverter() {}
    virtual ~PixelShuffleConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::pixel_shuffle};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor", {0, 0}}});
    }
};


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu