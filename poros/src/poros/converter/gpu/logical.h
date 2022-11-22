/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file logical.h
 @author Lin Xiao Chun (linxiaochun@baidu.com)
 @date 2022-02-17 18:32:23
 @brief

 */

#pragma once

#include <string>

//from pytorch
#include "torch/script.h"

#include "poros/converter/gpu/gpu_converter.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {

class AndConverter : public GpuConverter {
public:
    AndConverter() {}
    virtual ~AndConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor",
                };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::__and__,
                torch::jit::aten::__iand__,
                torch::jit::aten::bitwise_and,
                };
    }
};

class OrConverter : public GpuConverter {
public:
    OrConverter() {}
    virtual ~OrConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {
                "aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return  {torch::jit::aten::__or__,
                 torch::jit::aten::__ior__,
                 torch::jit::aten::bitwise_or,
        };
    }
};

class XorConverter : public GpuConverter {
public:
    XorConverter() {}
    virtual ~XorConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {
                "aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor",
                "aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::__xor__,
                torch::jit::aten::__ixor__,
                torch::jit::aten::bitwise_xor,
        };
    }
};

class NotConverter : public GpuConverter {
public:
    NotConverter() {}
    virtual ~NotConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //aten::bitwise_not(Tensor self) -> Tensor
    const std::vector<std::string> schema_string() {
        return {
                "aten::bitwise_not(Tensor self) -> Tensor",
        };
    }

    /** TODO: TO SUPPORT CONVERTERS BELLOW:
     *
     * **/
    const std::vector<torch::jit::NodeKind> node_kind() {
        return {
                torch::jit::aten::bitwise_not,
        };

    }
};

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
