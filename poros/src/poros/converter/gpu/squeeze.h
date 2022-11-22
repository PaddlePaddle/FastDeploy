/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file squeeze.h
* @author tianjinjin@baidu.com
* @date Wed Sep  1 11:19:13 CST 2021
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

class SqueezeConverter : public GpuConverter {
public:
    SqueezeConverter() {}
    virtual ~SqueezeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
                "aten::squeeze(Tensor(a) self) -> (Tensor(a))"};
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::squeeze(Tensor(a) self) -> Tensor(a)",
     * "aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)"
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::squeeze};
    }
};

class UnSqueezeConverter : public GpuConverter {
public:
    UnSqueezeConverter() {}
    virtual ~UnSqueezeConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::unsqueeze};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
