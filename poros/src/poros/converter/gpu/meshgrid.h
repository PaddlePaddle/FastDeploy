/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file meshgrid.h
* @author wangrui39@baidu.com
* @date Monday November 27 11:36:11 CST 2021
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

// DEPRECATED Use `lowering/fuse_meshgrid.h` to rewrite this op. This converter is no longer needed.
// Correspons to torch.meshgrid https://pytorch.org/docs/1.9.0/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
class MeshgridConverter : public GpuConverter {
public:
    MeshgridConverter() {}
    virtual ~MeshgridConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::meshgrid(Tensor[] tensors) -> Tensor[]"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::meshgrid};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::meshgrid(Tensor[] tensors) -> Tensor[]", {0, 0}}});
    }
};


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
