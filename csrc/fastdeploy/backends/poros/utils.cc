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

std::string AtType2String(const at::ScalarType& dtype) {
  std::string out;
  switch (dtype) {
    case at::kByte:
      out = "at::kByte";
      break;
    case at::kChar:
      out = "at::kChar";
      break;
    case at::kShort:
      out = "at::kShort";
      break;
    case at::kInt:
      out = "at::kInt";
      break;
    case at::kLong:
      out = "at::kLong";
      break;
    case at::kHalf:
      out = "at::kHalf";
      break;
    case at::kFloat:
      out = "at::kFloat";
      break;
    case at::kDouble:
      out = "at::kDouble";
      break;
    default:
      out = "at::UNKNOWN";
  }
  return out;
}

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
  FDERROR << "Unrecognized poros data type:" << AtType2String(poros_dtype) << "." << std::endl;
  return FDDataType::FP32;
}

at::Tensor CreatePorosValue(FDTensor& tensor, bool is_backend_cuda) {
  FDASSERT(tensor.device == Device::GPU || tensor.device == Device::CPU,
           "Only support tensor which device is CPU or GPU for PorosBackend.");
  auto data_type = GetPorosDtype(tensor.dtype);
  size_t numel = tensor.Numel();
  at::Tensor poros_value = std::move(at::empty(tensor.shape, {at::kCPU}).to(data_type).contiguous());
  if (data_type == at::kFloat) {
    memcpy(poros_value.data_ptr(), static_cast<void*>(tensor.Data()),
        numel * sizeof(float));
  } else if (data_type == at::kInt) {
    memcpy(poros_value.data_ptr(), static_cast<void*>(tensor.Data()),
        numel * sizeof(int32_t));
  } else if (data_type == at::kLong) {
    memcpy(poros_value.data_ptr(), static_cast<void*>(tensor.Data()),
        numel * sizeof(int64_t));
  } else if (data_type == at::kDouble) {
    memcpy(poros_value.data_ptr(), static_cast<void*>(tensor.Data()),
        numel * sizeof(double));
  } else {
    FDASSERT(false, "Unrecognized data type of " + Str(tensor.dtype) +
              " while calling PorosBackend::CreatePorosValue().");
  }
  // to cuda
  if (tensor.device == Device::GPU && is_backend_cuda) {
    poros_value = poros_value.to(at::kCUDA);
  }
  return poros_value;
}

void CopyTensorToCpu(const at::Tensor& tensor, FDTensor* fd_tensor) {
    std::cout << "test_wjj4444444" << std::endl;
    const auto data_type = tensor.scalar_type();
    std::vector<int64_t> shape;
    auto sizes = tensor.sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape.push_back(sizes[i]);
    }
    auto fd_dtype = GetFdDtype(data_type);
    // share memory
    fd_tensor->SetExternalData(shape, fd_dtype, tensor.data_ptr());
    return;
}

} // namespace fastdeploy