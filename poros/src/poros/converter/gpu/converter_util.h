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
* @file converter_util.h
* @author tianjinjin@baidu.com
* @date Thu Aug 12 10:50:28 CST 2021
* @brief
**/

#pragma once

#include <string>

#include "torch/script.h"
#include "NvInfer.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {


nvinfer1::ITensor* add_padding(TensorrtEngine* engine,
                                const torch::jit::Node* n, 
                                nvinfer1::ITensor* tensor,
                                int nDim,
                                bool trailing = true,
                                bool use_zeros = true);

nvinfer1::ITensor* add_unpadding(TensorrtEngine* engine,
                                const torch::jit::Node* n,
                                nvinfer1::ITensor* tensor,
                                int nDim,
                                bool trailing = true,
                                bool use_zeros = true);

nvinfer1::ILayer* add_elementwise(TensorrtEngine* engine,
                                nvinfer1::ElementWiseOperation op,
                                nvinfer1::ITensor* self,
                                nvinfer1::ITensor* other,
                                const std::string& name);

nvinfer1::ITensor* broadcast_itensor(TensorrtEngine* engine,
                                const torch::jit::Node* n,
                                nvinfer1::ITensor* tensor,
                                const int new_rank,
                                std::string name);

//If an ITensor is of a type not dtype, add an Identity layer to cast it to dtype
nvinfer1::ITensor* cast_itensor(TensorrtEngine* engine,
                                nvinfer1::ITensor* tensor, 
                                nvinfer1::DataType dtype);
                                
// 对nv shape tensor进行unsqueeze操作, 支持dim倒序
nvinfer1::ITensor* unsqueeze_nv_shapetensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input, 
                                    int dim);
// 对nv shape tensor进行squeeze操作, 支持dim倒序
// note: 使用前须检查 input[dim] == 1
nvinfer1::ITensor* squeeze_nv_shapetensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input, int dim);

// 对nv tensor进行unsqueeze操作
nvinfer1::ITensor* unsqueeze_itensor(TensorrtEngine* engine, 
                                    nvinfer1::ITensor* input,
                                    const std::vector<int>& axes);

//TODO: 添加对 nv tensor 进行squeeze操作
// nvinfer1::ITensor* squeeze_itensor(TensorrtEngine* engine,
//                                     nvinfer1::ITensor* input,
//                                     const std::vector<int>& axes);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
