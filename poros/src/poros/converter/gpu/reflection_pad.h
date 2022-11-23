/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file reflection_pad.h
* @author tianshaoqing@baidu.com
* @date Tue Aug 16 16:54:20 CST 2022
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

class ReflectionPadConverter : public GpuConverter {
public:
    ReflectionPadConverter() {}
    virtual ~ReflectionPadConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor",
                "aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor",
                };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::reflection_pad1d,
                torch::jit::aten::reflection_pad2d,
                };
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor", {1, 1}}});
        result &= assign_schema_attr_helper({{"aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor", {1, 1}}});
        return result;
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
