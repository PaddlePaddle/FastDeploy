/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file layer_norm.h
* @author tianjinjin@baidu.com
* @date Fri Aug 20 15:28:37 CST 2021
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

class LayerNormConverter : public GpuConverter {
public:
    LayerNormConverter() {}
    virtual ~LayerNormConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    virtual const std::vector<std::string> schema_string() {
        return {"aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"};
    }

    virtual const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::layer_norm};
    }

    bool assign_schema_attr() {
        bool result = true;
        result &= assign_schema_attr_helper({{"aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor", {1, 1}}});
        return result;
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
