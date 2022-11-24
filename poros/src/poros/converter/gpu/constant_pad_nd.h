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
* @file constant_pad_nd.h
* @author tianshaoqing@baidu.com
* @date Thur Dec 2 14:29:20 CST 2021
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

class ConstantPadNdConverter : public GpuConverter {
public:
    ConstantPadNdConverter() {}
    virtual ~ConstantPadNdConverter() {}

    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //DEPRECATED: 该实现方式内部采用的contat，在trt的profile阶段，会额外引入一些copy节点，导致性能变差。
    bool converter_old_version(TensorrtEngine* engine, const torch::jit::Node *node);

    const std::vector<std::string> schema_string() {
        return {"aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor"};
    }

    const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::constant_pad_nd};
    }

    bool assign_schema_attr() {
        return assign_schema_attr_helper({{"aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor", {1, 0}}});
    }

private:
    /** 
     * @brief 将pytorch组织的padding信息，转化成tensorrt可以接受的padding。
     * @param [in] engine : 略
     * @param [in] rank : 被padding的tensor的rank信息（也就是nbDims值）
     * @param [in] padding : pytorch序的padding信息，是 int[] 类型（注意: pytorch 的padding信息是从后往前的）
     * @param [out] start_tensor : 用于slice 的start tensor信息
     * @param [out] total_padding_tensor : padding 引入后每一维增多的size信息。
     * @return  bool
     * @retval true => succeed  false => failed
     * **/
    bool converter_padding(TensorrtEngine* engine,
                    int64_t rank,
                    const std::vector<int64_t>& padding,
                    nvinfer1::ITensor*& start_tensor,
                    nvinfer1::ITensor*& total_padding_tensor);
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
