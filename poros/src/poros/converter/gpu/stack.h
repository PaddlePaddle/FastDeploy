/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file stack.h
* @author tianjinjin@baidu.com
* @date Tue Sep  7 15:09:14 CST 2021
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

class StackConverter : public GpuConverter {
public:
    StackConverter() {}
    virtual ~StackConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
                "aten::vstack(Tensor[] tensors) -> Tensor"
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     * "aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::stack,
                torch::jit::aten::vstack,
                };
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", {1, 1}}});
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
