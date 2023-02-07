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
  int h = tensor.Shape()[0];
  int w = tensor.Shape()[1];
  int c = tensor.Shape()[2];

  nvcv::TensorDataStridedCuda::Buffer buf;
  buf.strides[3] = FDDataTypeSize(tensor.Dtype());
  buf.strides[2] = c * buf.strides[3];
  buf.strides[1] = w * buf.strides[2];
  buf.strides[0] = h * buf.strides[1];
  buf.basePtr = reinterpret_cast<NVCVByte*>(const_cast<void*>(tensor.Data()));

  nvcv::Tensor::Requirements req = nvcv::Tensor::CalcRequirements(
      batchsize, {w, h}, CreateCvCudaImageFormat(tensor.Dtype(), c));

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

nvcv::ImageWrapData CreateImageWrapData(const FDTensor& tensor) {
  FDASSERT(tensor.shape.size() == 3,
           "When create CVCUDA image from FD tensor,"
           "tensor shape should be 3-Dim, HWC layout");
  int h = tensor.Shape()[0];
  int w = tensor.Shape()[1];
  int c = tensor.Shape()[2];
  nvcv::ImageDataStridedCuda::Buffer buf;
  buf.numPlanes = 1;
  buf.planes[0].width = w;
  buf.planes[0].height = h;
  buf.planes[0].rowStride = w * c * FDDataTypeSize(tensor.Dtype());
  buf.planes[0].basePtr =
      reinterpret_cast<NVCVByte*>(const_cast<void*>(tensor.Data()));
  nvcv::ImageWrapData nvimg{nvcv::ImageDataStridedCuda{
      nvcv::ImageFormat{CreateCvCudaImageFormat(tensor.Dtype(), c)}, buf}};
  return nvimg;
}

void CreateCvCudaImageBatchVarShape(std::vector<FDTensor*>& tensors,
                                    nvcv::ImageBatchVarShape& img_batch) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    FDASSERT(tensors[i]->device == Device::GPU, "Tensor must on GPU.");
    img_batch.pushBack(CreateImageWrapData(*(tensors[i])));
  }
}
#endif

}  // namespace vision
}  // namespace fastdeploy
