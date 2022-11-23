/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file softmax.h
* @author tianjinjin@baidu.com
* @date Tue Aug 24 17:15:33 CST 2021
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

class SoftmaxConverter : public GpuConverter {
public:
    SoftmaxConverter() {}
    virtual ~SoftmaxConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"};
    }
    
    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
     **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::softmax};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
