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

#include "fastdeploy/vision/common/processors/cvcuda_utils.h"

namespace fastdeploy {
namespace vision {

#ifdef ENABLE_CVCUDA
nvcv::ImageFormat CreateCvCudaImageFormat(FDDataType type, int channel) {
  FDASSERT(channel == 1 || channel == 3 || channel == 4,
           "Only support channel be 1/3/4 in CV-CUDA.");
  if (type == FDDataType::UINT8) {
    if (channel == 1) {
      return nvcv::FMT_U8;
    } else if (channel == 3) {
      return nvcv::FMT_BGR8;
    } else {
      return nvcv::FMT_BGRA8;
    }
  } else if (type == FDDataType::FP32) {
    if (channel == 1) {
      return nvcv::FMT_F32;
    } else if (channel == 3) {
      return nvcv::FMT_BGRf32;
    } else {
      return nvcv::FMT_BGRAf32;
    }
  }
  FDASSERT(false, "Data type of %s is not supported.", Str(type).c_str());
  return nvcv::FMT_BGRf32;
}

nvcv::TensorWrapData CreateCvCudaTensorWrapData(const FDTensor& tensor) {
  FDASSERT(tensor.shape.size() == 3,
           "When create CVCUDA tensor from FD tensor,"
           "tensor shape should be 3-Dim, HWC layout");
  int batchsize = 1;

  nvcv::TensorDataStridedCuda::Buffer buf;
  buf.strides[3] = FDDataTypeSize(tensor.Dtype());
  buf.strides[2] = tensor.shape[2] * buf.strides[3];
  buf.strides[1] = tensor.shape[1] * buf.strides[2];
  buf.strides[0] = tensor.shape[0] * buf.strides[1];
  buf.basePtr = reinterpret_cast<NVCVByte*>(const_cast<void*>(tensor.Data()));

  nvcv::Tensor::Requirements req = nvcv::Tensor::CalcRequirements(
      batchsize, {tensor.shape[1], tensor.shape[0]},
      CreateCvCudaImageFormat(tensor.Dtype(), tensor.shape[2]));

  nvcv::TensorDataStridedCuda tensor_data(
      nvcv::TensorShape{req.shape, req.rank, req.layout},
      nvcv::DataType{req.dtype}, buf);
  return nvcv::TensorWrapData(tensor_data);
}

void* GetCvCudaTensorDataPtr(const nvcv::TensorWrapData& tensor) {
  auto data =
      dynamic_cast<const nvcv::ITensorDataStridedCuda*>(tensor.exportData());
  return reinterpret_cast<void*>(data->basePtr());
}
#endif

}  // namespace vision
}  // namespace fastdeploy
