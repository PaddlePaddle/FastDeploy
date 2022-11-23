/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file coercion.h
* @author wangrui39@baidu.com
* @date Fri May 13 11:36:11 CST 2022
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

class CoercionConverter : public GpuConverter {
public:
    CoercionConverter() {}
    virtual ~CoercionConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::Int.float(float a) -> (int)", 
                "aten::Int.Tensor(Tensor a) -> (int)"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::Int};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::Int.float(float a) -> (int)", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::Int.Tensor(Tensor a) -> (int)", {1, 1}}});
        return result;
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
