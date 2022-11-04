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
#include "fastdeploy/vision/common/processors/utils.h"

namespace fastdeploy {
namespace vision {

void* Mat::Data() {
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return fcv_mat.data();
#else
    FDASSERT(false,
             "FastDeploy didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  }
  return cpu_mat.ptr();
}

void Mat::ShareWithTensor(FDTensor* tensor) {
  tensor->SetExternalData({Channels(), Height(), Width()}, Type(), Data());
  tensor->device = Device::CPU;
  if (layout == Layout::HWC) {
    tensor->shape = {Height(), Width(), Channels()};
  }
}

bool Mat::CopyToTensor(FDTensor* tensor) {
  int total_bytes = Height() * Width() * Channels() * FDDataTypeSize(Type());
  if (total_bytes != tensor->Nbytes()) {
    FDERROR << "While copy Mat to Tensor, requires the memory size be same, "
               "but now size of Tensor = "
            << tensor->Nbytes() << ", size of Mat = " << total_bytes << "."
            << std::endl;
    return false;
  }
  memcpy(tensor->MutableData(), Data(), total_bytes);
  return true;
}

void Mat::PrintInfo(const std::string& flag) {
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Scalar mean = fcv::mean(fcv_mat);
    std::cout << flag << ": "
              << "DataType=" << Type() << ", "
              << "Channel=" << Channels() << ", "
              << "Height=" << Height() << ", "
              << "Width=" << Width() << ", "
              << "Mean=";
    for (int i = 0; i < Channels(); ++i) {
      std::cout << mean[i] << " ";
    }
    std::cout << std::endl;
#else
    FDASSERT(false,
             "FastDeploy didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  } else {
    cv::Scalar mean = cv::mean(cpu_mat);
    std::cout << flag << ": "
              << "DataType=" << Type() << ", "
              << "Channel=" << Channels() << ", "
              << "Height=" << Height() << ", "
              << "Width=" << Width() << ", "
              << "Mean=";
    for (int i = 0; i < Channels(); ++i) {
      std::cout << mean[i] << " ";
    }
    std::cout << std::endl;
  }
}

FDDataType Mat::Type() {
  int type = -1;
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return FlyCVDataTypeToFD(fcv_mat.type());
#else
    FDASSERT(false,
             "FastDeploy didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  }
  return OpenCVDataTypeToFD(cpu_mat.type());
}

std::ostream& operator<<(std::ostream& out, const ProcLib& p) {
  switch (p) {
    case ProcLib::DEFAULT:
      out << "ProcLib::DEFAULT";
      break;
    case ProcLib::OPENCV:
      out << "ProcLib::OPENCV";
      break;
    case ProcLib::FLYCV:
      out << "ProcLib::FLYCV";
      break;
    default:
      FDASSERT(false, "Unknow type of ProcLib.");
  }
  return out;
}

Mat Mat::Create(const FDTensor& tensor) {
  if (DefaultProcLib::default_lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat fcv_mat = CreateZeroCopyFlyCVMatFromTensor(tensor);
    Mat mat = Mat(fcv_mat);
    return mat;
#else
    FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
  }
  cv::Mat ocv_mat = CreateZeroCopyOpenCVMatFromTensor(tensor);
  Mat mat = Mat(ocv_mat);
  return mat;
}

Mat Mat::Create(const FDTensor& tensor, ProcLib lib) {
  if (lib == ProcLib::DEFAULT) {
    return Create(tensor);
  }
  if (lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat fcv_mat = CreateZeroCopyFlyCVMatFromTensor(tensor);
    Mat mat = Mat(fcv_mat);
    return mat;
#else
    FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
  } 
  cv::Mat ocv_mat = CreateZeroCopyOpenCVMatFromTensor(tensor);
  Mat mat = Mat(ocv_mat);
  return mat;
}

Mat Mat::Create(int height, int width, int channels,
                FDDataType type, void* data) {
  if (DefaultProcLib::default_lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat fcv_mat = CreateZeroCopyFlyCVMatFromBuffer(
      height, width, channels, type, data);
    Mat mat = Mat(fcv_mat);
    return mat;
#else
    FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
  }
  cv::Mat ocv_mat = CreateZeroCopyOpenCVMatFromBuffer(
      height, width, channels, type, data);
  Mat mat = Mat(ocv_mat);
  return mat;    
}

Mat Mat::Create(int height, int width, int channels,
                FDDataType type, void* data,
                ProcLib lib) {
  if (lib == ProcLib::DEFAULT) {
    return Create(height, width, channels, type, data);
  }                  
  if (lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat fcv_mat = CreateZeroCopyFlyCVMatFromBuffer(
      height, width, channels, type, data);
    Mat mat = Mat(fcv_mat);
    return mat;
#else
    FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
  } 
  cv::Mat ocv_mat = CreateZeroCopyOpenCVMatFromBuffer(
      height, width, channels, type, data);
  Mat mat = Mat(ocv_mat);
  return mat;    
}

}  // namespace vision
}  // namespace fastdeploy
