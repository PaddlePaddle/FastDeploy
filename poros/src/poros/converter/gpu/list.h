/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
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
