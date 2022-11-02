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
#include "fastdeploy/vision/common/processors/utils.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace vision {

void* Mat::Data() {
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return fcv_mat.data();
#else
    FDASSERT(false, "FastDeploy didn't compile with FalconCV, but met data type with fcv::Mat.");
#endif
  }
  return cpu_mat.ptr();
}

void Mat::ShareWithTensor(FDTensor* tensor) {
  tensor->SetExternalData({Channels(), Height(), Width()}, Type(),
                          Data());
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
    FDASSERT(false, "FastDeploy didn't compile with FalconCV, but met data type with fcv::Mat.");
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
    return FalconCVDataTypeToFD(fcv_mat.type());
#else
    FDASSERT(false, "FastDeploy didn't compile with FalconCV, but met data type with fcv::Mat.");
#endif
  }
  return OpenCVDataTypeToFD(cpu_mat.type());
}

Mat CreateFromTensor(const FDTensor& tensor) {
  int type = tensor.dtype;
  cv::Mat temp_mat;
  FDASSERT(tensor.shape.size() == 3,
           "When create FD Mat from tensor, tensor shape should be 3-Dim, HWC "
           "layout");
  int64_t height = tensor.shape[0];
  int64_t width = tensor.shape[1];
  int64_t channel = tensor.shape[2];
  switch (type) {
    case FDDataType::UINT8:
      temp_mat = cv::Mat(height, width, CV_8UC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT8:
      temp_mat = cv::Mat(height, width, CV_8SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT16:
      temp_mat = cv::Mat(height, width, CV_16SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::INT32:
      temp_mat = cv::Mat(height, width, CV_32SC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::FP32:
      temp_mat = cv::Mat(height, width, CV_32FC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    case FDDataType::FP64:
      temp_mat = cv::Mat(height, width, CV_64FC(channel),
                         const_cast<void*>(tensor.Data()));
      break;

    default:
      FDASSERT(
          false,
          "Tensor type %d is not supported While calling CreateFromTensor.",
          type);
      break;
  }
  Mat mat = Mat(temp_mat);
  return mat;
}

std::ostream& operator<<(std::ostream& out,const ProcLib& p) {
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




}  // namespace vision
}  // namespace fastdeploy
