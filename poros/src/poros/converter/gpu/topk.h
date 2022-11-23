/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file topk.h
* @author tianjinjin@baidu.com
* @date Tue Sep  7 14:29:20 CST 2021
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

class TopkConverter : public GpuConverter {
public:
    TopkConverter() {}
    virtual ~TopkConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::topk,
                };
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
