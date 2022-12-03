// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
