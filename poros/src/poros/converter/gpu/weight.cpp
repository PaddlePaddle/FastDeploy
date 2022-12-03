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
* @file weight.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug  6 14:17:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/weight.h"
#include "poros/engine/trtengine_util.h"

namespace baidu {
namespace mirana {
namespace poros {

Weights::Weights() {
    this->inputs_num = 0;
    this->outputs_num = 0;
    this->data.type = nvinfer1::DataType::kFLOAT;
    this->data.values = nullptr;
    this->data.count = 0;
}

Weights::Weights(at::Tensor tensor) {
    POROS_CHECK((tensor.sizes().size() <= nvinfer1::Dims::MAX_DIMS), 
        "given tensor is outof max_dims");
        
    if (tensor.scalar_type() == c10::ScalarType::Long) {
        LOG(WARNING) << "Weights meets c10::ScalarType::Long tensor type, change this to c10::ScalarType::Int. "
                << "Attention: this may leed to percision change";
        tensor = tensor.to(at::ScalarType::Int);
    }

    this->shape = sizes_to_nvdim(tensor.sizes());
    //TODO: CHECK this bias info. 
    this->inputs_num = (tensor.sizes().size() > 1) ? tensor.sizes()[1] : tensor.sizes()[0];
    this->outputs_num = tensor.sizes()[0];

    if (tensor.sizes().size() > 2) {
        this->kernel_shape.nbDims = tensor.sizes().size() - 2;
        for (size_t i = 2; i < tensor.sizes().size(); i++) {
            this->kernel_shape.d[i - 2] = tensor.sizes()[i];
        }
    } else {
        this->kernel_shape.nbDims = 1;
        this->kernel_shape.d[0] = 1;
    }

    auto t_cpu = tensor.to(at::kCPU);
    t_cpu = t_cpu.contiguous();

    auto t_type = c10::optTypeMetaToScalarType(t_cpu.dtype());
    POROS_CHECK(t_type.has_value(), "unsupported datatype");
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

    this->data.type = dtype;
    this->data.count = t_cpu.numel();
    this->data.values = buf;

}

std::ostream& operator<<(std::ostream& os, const Weights& w) {
  os << "Weights: " << w.shape
     << "\n    Number of input maps: " << w.inputs_num
     << "\n    Number of output maps: " << w.outputs_num
     << "\n    Element shape: [";
  for (int i = 0; i < w.kernel_shape.nbDims; i++) {
    os << w.kernel_shape.d[i];
    if (i + 1 < w.kernel_shape.nbDims) {
      os << ',';
    }
  }
  os << ']';
  return os;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
