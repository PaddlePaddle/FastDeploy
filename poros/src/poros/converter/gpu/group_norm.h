/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file group_norm.h
* @author tianshaoqing@baidu.com
* @date Fri Jan 21 15:28:37 CST 2022
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

class GroupNormConverter : public GpuConverter {
public:
    GroupNormConverter() {}
    virtual ~GroupNormConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    bool converter_old(TensorrtEngine* engine, const torch::jit::Node *node);

    virtual const std::vector<std::string> schema_string() {
        return {"aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"};
    }

    virtual const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::group_norm};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
