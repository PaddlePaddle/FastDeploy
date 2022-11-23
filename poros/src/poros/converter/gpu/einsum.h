/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file einsum.h
* @author tianshaoqing@baidu.com
* @date Wed Jul 06 11:24:51 CST 2022
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

class EinsumConverter : public GpuConverter {
public:
    EinsumConverter() {}
    virtual ~EinsumConverter() {}
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);
    const std::vector<std::string> schema_string() {
        return {"aten::einsum(str equation, Tensor[] tensors) -> (Tensor)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::einsum};
    }

    // mark: einsum动态的规则比较复杂，是根据equation情况来定的，先禁止dy
    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::einsum(str equation, Tensor[] tensors) -> (Tensor)", {0, 0}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
