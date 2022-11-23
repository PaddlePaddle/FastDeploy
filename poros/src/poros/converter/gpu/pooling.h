/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file pooling.h
* @author tianjinjin@baidu.com
* @date Tue Aug 17 22:57:03 CST 2021
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

class PoolingConverter : public GpuConverter {
public:
    PoolingConverter() {}
    virtual ~PoolingConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor",
                "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
                "aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor",
                "aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
                "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor",
                "aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"
            };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::max_pool1d,
                torch::jit::aten::avg_pool1d,
                torch::jit::aten::max_pool2d,
                torch::jit::aten::avg_pool2d,
                torch::jit::aten::max_pool3d,
                torch::jit::aten::avg_pool3d};
    }
};


class AdaptivePoolingConverter : public GpuConverter {
public:
    AdaptivePoolingConverter() {}
    virtual ~AdaptivePoolingConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor",
                "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
                "aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)"
            };
    }

    /** TODO: TRY TO SUPPORT SCHEMA PATTERNS BELLOW:
     * aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
     * aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
     * aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
     **/

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::adaptive_avg_pool1d,
                torch::jit::aten::adaptive_avg_pool2d,
                torch::jit::aten::adaptive_max_pool2d
            };
    }
};



}  // namespace poros 
}  // namespace mirana
}  // namespace baidu