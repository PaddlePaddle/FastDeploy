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
* @file aten_trt_util.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug  6 14:17:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/aten_trt_util.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/macros.h"

namespace baidu {
namespace mirana {
namespace poros {

//将torchscript中的at::tensor转变成tensorrt中的weights结构
bool at_tensor_to_trt_weignts(at::Tensor tensor, nvinfer1::Weights& weight) {
    POROS_CHECK_TRUE((tensor.sizes().size() <= nvinfer1::Dims::MAX_DIMS), 
        "given tensor is outof max_dims");

    /*
    auto shape = sizes_to_nvdim(tensor.sizes());
    //TODO: CHECK this bias info. 
    int64_t inputs_num = (tensor.sizes().size() > 1) ? tensor.sizes()[1] : tensor.sizes()[0];
    int64_t outputs_num = tensor.sizes()[0];

    nvinfer1::Dims kernel_shape;
    if (tensor.sizes().size() > 2) {
        kernel_shape.nbDims = tensor.sizes().size() - 2;
        for (size_t i = 2; i < tensor.sizes().size(); i++) {
            kernel_shape.d[i - 2] = tensor.size()[i];
        }
    } else {
        kernal_shape.nbdims = 1;
        kernal_shape.d[0] = 1;
    }*/

    auto t_cpu = tensor.to(at::kCPU);
    t_cpu = t_cpu.contiguous();

    auto t_type = c10::optTypeMetaToScalarType(t_cpu.dtype());
    POROS_CHECK_TRUE(t_type.has_value(), "unsupported datatype");
    //TODO: may be failed here
    auto dtype = attype_to_nvtype(t_type.value());

    void* buf = nullptr;
    if (dtype == nvinfer1::DataType::kFLOAT) {
        buf = malloc(t_cpu.numel() * sizeof(float));
        memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(float));
    } else if (dtype == nvinfer1::DataType::kHALF) {
        buf = malloc(t_cpu.numel() * (sizeof(float) / 2));
        memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * (sizeof(float) / 2));
    } else if (dtype == nvinfer1::DataType::kINT8) {
        buf = malloc(t_cpu.numel() * sizeof(char));
        memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(char));
    } else if (dtype == nvinfer1::DataType::kINT32) {
        buf = malloc(t_cpu.numel() * sizeof(int));
        memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(int));
    } else if (dtype == nvinfer1::DataType::kBOOL) {
        buf = malloc(t_cpu.numel() * sizeof(bool));
        memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(bool));
    }

    weight.type = dtype;
    weight.count = t_cpu.numel();
    weight.values = buf;

    return true;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
