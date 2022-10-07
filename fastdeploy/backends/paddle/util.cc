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

#include "fastdeploy/backends/paddle/paddle_backend.h"
#include "fastdeploy/core/float16.h"

namespace fastdeploy {
paddle_infer::PlaceType ConvertFDDeviceToPlace(Device device) {
  if (device == Device::GPU) {
    return paddle_infer::PlaceType::kGPU;
  }
  return paddle_infer::PlaceType::kCPU;
}

void ShareTensorFromFDTensor(paddle_infer::Tensor* tensor,
                             FDTensor& fd_tensor) {
  std::vector<int> shape(fd_tensor.shape.begin(), fd_tensor.shape.end());
  tensor->Reshape(shape);
  auto place = ConvertFDDeviceToPlace(fd_tensor.device);
  if (fd_tensor.dtype == FDDataType::FP32) {
    tensor->ShareExternalData(static_cast<const float*>(fd_tensor.Data()),
                              shape, place);
    return;
  } else if (fd_tensor.dtype == FDDataType::INT32) {
    tensor->ShareExternalData(static_cast<const int32_t*>(fd_tensor.Data()),
                              shape, place);
    return;
  } else if (fd_tensor.dtype == FDDataType::INT64) {
    tensor->ShareExternalData(static_cast<const int64_t*>(fd_tensor.Data()),
                              shape, place);
    return;
  } else if (fd_tensor.dtype == FDDataType::UINT8) {
    tensor->ShareExternalData(static_cast<const uint8_t*>(fd_tensor.Data()),
                              shape, paddle_infer::PlaceType::kCPU);
    return;
  }
  FDASSERT(false, "Unexpected data type(%s) while infer with PaddleBackend.",
           Str(fd_tensor.dtype).c_str());
}

void CopyTensorToCpu(std::unique_ptr<paddle_infer::Tensor>& tensor,
                     FDTensor* fd_tensor) {
  auto fd_dtype = PaddleDataTypeToFD(tensor->type());
  std::vector<int64_t> shape;
  auto tmp_shape = tensor->shape();
  shape.assign(tmp_shape.begin(), tmp_shape.end());
  fd_tensor->Allocate(shape, fd_dtype, tensor->name());
  if (fd_tensor->dtype == FDDataType::FP32) {
    tensor->CopyToCpu(static_cast<float*>(fd_tensor->MutableData()));
    return;
  } else if (fd_tensor->dtype == FDDataType::INT32) {
    tensor->CopyToCpu(static_cast<int32_t*>(fd_tensor->MutableData()));
    return;
  } else if (fd_tensor->dtype == FDDataType::INT64) {
    tensor->CopyToCpu(static_cast<int64_t*>(fd_tensor->MutableData()));
    return;
  }
  FDASSERT(false, "Unexpected data type(%s) while infer with PaddleBackend.",
           Str(fd_tensor->dtype).c_str());
}

FDDataType PaddleDataTypeToFD(const paddle_infer::DataType& dtype) {
  auto fd_dtype = FDDataType::FP32;
  if (dtype == paddle_infer::FLOAT32) {
    fd_dtype = FDDataType::FP32;
  } else if (dtype == paddle_infer::INT64) {
    fd_dtype = FDDataType::INT64;
  } else if (dtype == paddle_infer::INT32) {
    fd_dtype = FDDataType::INT32;
  } else if (dtype == paddle_infer::UINT8) {
    fd_dtype = FDDataType::UINT8;
  } else if (dtype == paddle_infer::INT8) {
    fd_dtype = FDDataType::INT8;
  } else if (dtype == paddle_infer::FLOAT16) {
    fd_dtype = FDDataType::FP16;
  } else {
    FDASSERT(
        false,
        "Unexpected data type: %d while call CopyTensorToCpu in PaddleBackend.",
        int(dtype));
  }
  return fd_dtype;
}

}  // namespace fastdeploy
