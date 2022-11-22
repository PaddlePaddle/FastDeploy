/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file aten_eval.h
* @author tianjinjin@baidu.com
* @date Mon Aug 16 12:26:28 CST 2021
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

class AppendConverter : public GpuConverter {
public:
    AppendConverter() {}
    virtual ~AppendConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    const std::vector<std::string> schema_string() {
        return {"aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))" };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::append};
    }
};

class GetitemConverter : public GpuConverter {
public:
    GetitemConverter() {}
    virtual ~GetitemConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::__getitem__.t(t[](a) list, int idx) -> (t(*))
    const std::vector<std::string> schema_string() {
        return {"aten::__getitem__.t(t[](a) list, int idx) -> (t(*))"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::__getitem__};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::__getitem__.t(t[](a) list, int idx) -> (t(*))", {1, 1}}});
    }
};

class SetitemConverter : public GpuConverter {
public:
    SetitemConverter() {}
    virtual ~SetitemConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::_set_item.t(t[](a!) l, int idx, t(b -> *) el) -> (t[](a!))
    const std::vector<std::string> schema_string() {
        return {"aten::_set_item.t(t[](a!) l, int idx, t(b -> *) el) -> (t[](a!))"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::_set_item};
    }
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
