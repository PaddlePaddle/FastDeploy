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
* @file convolution.h
* @author tianjinjin@baidu.com
* @date Wed Aug 11 16:00:26 CST 2021
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

class ConvolutionConverter : public GpuConverter {
public:
    ConvolutionConverter() {}
    virtual ~ConvolutionConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);
    // bool converter(TensorrtEngine* engine,
    //             const std::vector<const torch::jit::Value*> inputs, 
    //             const std::vector<const torch::jit::Value*> outputs);

    const std::vector<std::string> schema_string() {
        return {"aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
        "aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
        "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
        "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
        "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor"
        };
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::_convolution,
                torch::jit::aten::conv1d,
                torch::jit::aten::conv2d,
                torch::jit::aten::conv3d};
    }
};


}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
