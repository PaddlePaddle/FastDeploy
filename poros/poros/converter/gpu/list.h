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
* @file list.h
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

class ListConstructConverter : public GpuConverter {
public:
    ListConstructConverter() {}
    virtual ~ListConstructConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::ListConstruct kind node has no schema
    const std::vector<std::string> schema_string() {
        return {};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::ListConstruct};
    }
};

class ListUnpackConverter : public GpuConverter {
public:
    ListUnpackConverter() {}
    virtual ~ListUnpackConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::ListUnpack kind node has no schema
    const std::vector<std::string> schema_string() {
        return {};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::prim::ListUnpack};
    }
};

class ListConverter : public GpuConverter {
public:
    ListConverter() {}
    virtual ~ListConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //prim::List kind node has no schema
    const std::vector<std::string> schema_string() {
        return {};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::list};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
