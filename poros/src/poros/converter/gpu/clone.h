/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file clone.h
* @author tianshaoqing@baidu.com
* @date Tue Nov 23 12:26:28 CST 2021
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

class CloneConverter : public GpuConverter {
public:
    CloneConverter() {}
    virtual ~CloneConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    const std::vector<std::string> schema_string() {
        return {"aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::clone};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
