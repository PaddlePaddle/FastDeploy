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
* @file reduce.h
* @author tianjinjin@baidu.com
* @date Fri Aug 27 10:18:24 CST 2021
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

class MeanConverter : public GpuConverter {
public:
    MeanConverter() {}
    virtual ~MeanConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
                "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
     * "aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
     * "aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::mean};
    }
};

class SumConverter : public GpuConverter {
public:
    SumConverter() {}
    virtual ~SumConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
                "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
     * "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
     * "aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::sum};
    }
};

class ProdConverter : public GpuConverter {
public:
    ProdConverter() {}
    virtual ~ProdConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
                "aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
     * "aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
     * "aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::prod};
    }
};

class MaxMinConverter : public GpuConverter {
public:
    MaxMinConverter() {}
    virtual ~MaxMinConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::max(Tensor self) -> Tensor",
                "aten::min(Tensor self) -> Tensor",
                "aten::max.other(Tensor self, Tensor other) -> Tensor",
                "aten::min.other(Tensor self, Tensor other) -> Tensor",
                "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
                "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",};
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)",
     * "aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
     * "aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)",
     * "aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
     * 
     * "aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)",
     * "aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
     * "aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)",
     * "aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::max,
                torch::jit::aten::min};
    }
};

class ArgmaxArgminConverter : public GpuConverter {
public:
    ArgmaxArgminConverter() {}
    virtual ~ArgmaxArgminConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::argmax(Tensor self, int dim, bool keepdim=False) -> (Tensor)",
                "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)",
                "aten::argmin(Tensor self, int dim, bool keepdim=False) -> (Tensor)",
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::argmax,
                torch::jit::aten::argmin};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
