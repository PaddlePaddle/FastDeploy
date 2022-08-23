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

#include "fastdeploy/backends/poros/poros_backend.h"

namespace fastdeploy {

at::ScalarType GetPorosDtype(const FDDataType& fd_dtype) {
  if (fd_dtype == FDDataType::FP32) {
    return at::kFloat;
  } else if (fd_dtype == FDDataType::FP64) {
    return at::kDouble;
  } else if (fd_dtype == FDDataType::INT32) {
    return at::kInt;
  } else if (fd_dtype == FDDataType::INT64) {
    return at::kLong;
  }
  FDERROR << "Unrecognized fastdeply data type:" << Str(fd_dtype) << "."
          << std::endl;
  return at::kFloat;
}

FDDataType GetFdDtype(const at::ScalarType& poros_dtype) {
  if (poros_dtype == at::kFloat) {
    return FDDataType::FP32;
  } else if (poros_dtype == at::kDouble) {
    return FDDataType::FP64;
  } else if (poros_dtype == at::kInt) {
    return FDDataType::INT32;
  } else if (poros_dtype == at::kLong) {
    return FDDataType::INT64;
  }
  FDERROR << "Unrecognized poros data type:" << poros_dtype << "." << std::endl;
  return FDDataType::FP32;
}

at::Tensor CreatePorosValue(FDTensor& tensor, bool is_backend_cuda) {
  FDASSERT(tensor.device == Device::GPU || tensor.device == Device::CPU,
           "Only support tensor which device is CPU or GPU for PorosBackend.");
  if (tensor.device == Device::GPU && is_backend_cuda) {
      at::tensor poros_value = std::move(at::empty(tensor.shape, {at::kCUDA}).to(GetPorosDtype(tensor.dtype)).contiguous()); 
      poros_value.data_ptr() = tensor.Data();
      return poros_value;
  }
  at::tensor poros_value = std::move(at::empty(tensor.shape, {at::kCPU}).to(GetPorosDtype(tensor.dtype)).contiguous());
  poros_value.data_ptr() = tensor.Data();
  return poros_value;
}

void CopyTensorToCpu(const at::Tensor& tensor, FDTensor* fd_tensor) {
    const auto data_type = tensor.scalar_type();
    std::vector<int64_t> shape;
    auto sizes = tensor.sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape.push_back(sizes[i]);
    }
    fd_tensor->shape = shape;
    size_t numel = tensor.numel();

    if (data_type == at::kFloat) {
        fd_tensor->data.resize(numel * sizeof(float));
        memcpy(static_cast<void*>(fd_tensor->Data()), tensor.data_ptr(),
            numel * sizeof(float));
        fd_tensor->dtype = FDDataType::FP32;
    } else if (data_type == at::kInt) {
        fd_tensor->data.resize(numel * sizeof(int32_t));
        memcpy(static_cast<void*>(fd_tensor->Data()), tensor.data_ptr(),
            numel * sizeof(int32_t));
        fd_tensor->dtype = FDDataType::INT32;
    } else if (data_type == at::kLong) {
        fd_tensor->data.resize(numel * sizeof(int64_t));
        memcpy(static_cast<void*>(fd_tensor->Data()), tensor.data_ptr(),
            numel * sizeof(int64_t));
        fd_tensor->dtype = FDDataType::INT64;
    } else if (data_type == at::kDouble) {
        fd_tensor->data.resize(numel * sizeof(double));
        memcpy(static_cast<void*>(fd_tensor->Data()), tensor.data_ptr(),
            numel * sizeof(double));
        fd_tensor->dtype = FDDataType::FP64;
    } else {
        FDASSERT(false, "Unrecognized data type of " + std::to_string(data_type) +
                            " while calling OrtBackend::CopyToCpu().");
    }
}


}