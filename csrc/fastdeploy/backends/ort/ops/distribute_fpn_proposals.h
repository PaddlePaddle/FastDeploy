// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>

#ifndef NON_64_PLATFORM
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace fastdeploy {

struct DistributeFpnProposalsKernel {
 protected:
  Ort::CustomOpApi ort_;

 public:
  int64_t max_level;
  int64_t min_level;
  int64_t refer_level;
  int64_t refer_scale;
  int64_t pixel_offset;

  DistributeFpnProposalsKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    GetAttribute(info);
  }

  void GetAttribute(const OrtKernelInfo* info);

  void Compute(OrtKernelContext* context);
};

struct DistributeFpnProposalsWithLevelOp
    : Ort::CustomOpBase<DistributeFpnProposalsWithLevelOp, DistributeFpnProposalsKernel> {
  int num_level = 4;
  DistributeFpnProposalsWithLevelOp(int _num_level) {
    num_level = _num_level;
  }

  virtual void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new DistributeFpnProposalsKernel(api, info);
  }

  virtual const char* GetName() const { return ("DistributeFpnProposalsWithLevel" + std::to_string(num_level)).c_str(); }

  virtual size_t GetInputTypeCount() const { return 2; }

  virtual ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 1) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  virtual size_t GetOutputTypeCount() const {
     return num_level * 2 + 1;
  }

  virtual ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index >= num_level) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  virtual const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
};

}  // namespace fastdeploy

#endif
