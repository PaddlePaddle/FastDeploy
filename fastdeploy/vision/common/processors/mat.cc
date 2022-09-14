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
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/utils/utils.h"
namespace fastdeploy {
namespace vision {

#ifdef ENABLE_OPENCV_CUDA
cv::cuda::GpuMat* Mat::GetGpuMat() {
  if (device == Device::CPU) {
    gpu_mat.upload(cpu_mat);
  }
  return &gpu_mat;
}
#endif

cv::Mat* Mat::GetCpuMat() {
#ifdef ENABLE_OPENCV_CUDA
  if (device == Device::GPU) {
    gpu_mat.download(cpu_mat);
  }
#endif
  return &cpu_mat;
}

void Mat::ShareWithTensor(FDTensor* tensor) {
  if (device == Device::GPU) {
#ifdef ENABLE_OPENCV_CUDA
    tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                            GetGpuMat()->ptr());
    tensor->device = Device::GPU;
#endif
  } else {
    tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                            GetCpuMat()->ptr());
    tensor->device = Device::CPU;
  }
  if (layout == Layout::HWC) {
    tensor->shape = {Height(), Width(), Channels()};
  }
}

bool Mat::CopyToTensor(FDTensor* tensor) {
  cv::Mat* im = GetCpuMat();
  int total_bytes = im->total() * im->elemSize();
  if (total_bytes != tensor->Nbytes()) {
    FDERROR << "While copy Mat to Tensor, requires the memory size be same, "
               "but now size of Tensor = "
            << tensor->Nbytes() << ", size of Mat = " << total_bytes << "."
            << std::endl;
    return false;
  }
  memcpy(tensor->MutableData(), im->ptr(), im->total() * im->elemSize());
  return true;
}

void Mat::PrintInfo(const std::string& flag) {
  cv::Mat* im = GetCpuMat();
  cv::Scalar mean = cv::mean(*im);
  std::cout << flag << ": "
            << "Channel=" << Channels() << ", height=" << Height()
            << ", width=" << Width() << ", mean=";
  for (int i = 0; i < Channels(); ++i) {
    std::cout << mean[i] << " ";
  }
  std::cout << std::endl;
}

FDDataType Mat::Type() {
  int type = -1;
  if (device == Device::GPU) {
#ifdef ENABLE_OPENCV_CUDA
    type = gpu_mat.type();
#endif
  } else {
    type = cpu_mat.type();
  }
  if (type < 0) {
    FDASSERT(false,
             "While calling Mat::Type(), get negative value, which is not "
             "expected!.");
  }
  type = type % 8;
  if (type == 0) {
    return FDDataType::UINT8;
  } else if (type == 1) {
    return FDDataType::INT8;
  } else if (type == 2) {
    FDASSERT(false,
             "While calling Mat::Type(), get UINT16 type which is not "
             "supported now.");
  } else if (type == 3) {
    return FDDataType::INT16;
  } else if (type == 4) {
    return FDDataType::INT32;
  } else if (type == 5) {
    return FDDataType::FP32;
  } else if (type == 6) {
    return FDDataType::FP64;
  } else {
    FDASSERT(
        false,
        "While calling Mat::Type(), get type = %d, which is not expected!.",
        type);
  }
}

}  // namespace vision
}  // namespace fastdeploy
