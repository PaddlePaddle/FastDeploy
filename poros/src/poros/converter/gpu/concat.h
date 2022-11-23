/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file concat.h
* @author tianjinjin@baidu.com
* @date Tue Jul 27 11:24:21 CST 2021
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

//TODO: there is a concat_opt.cpp in torchscript. check it.
class ConcatConverter : public GpuConverter {
public:
    ConcatConverter() {}
    virtual ~ConcatConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::cat};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", {1, 1}}});
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
