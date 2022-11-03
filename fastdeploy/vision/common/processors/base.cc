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

#include "fastdeploy/vision/common/processors/base.h"

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace vision {

ProcLib Processor::default_lib = ProcLib::DEFAULT;

bool Processor::operator()(Mat* mat, ProcLib lib) {
  ProcLib target = lib;
  if (lib == ProcLib::DEFAULT) {
    target = default_lib;
  }
  if (target == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return ImplByFlyCV(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with FlyCV.");
#endif
  }
  // DEFAULT & OPENCV
  return ImplByOpenCV(mat);
}

void EnableFlyCV() {
#ifdef ENABLE_FLYCV
  Processor::default_lib = ProcLib::FLYCV;
  FDINFO << "Will change to use image processing library "
         << Processor::default_lib << std::endl;
#else
  FDWARNING << "FastDeploy didn't compile with FlyCV, "
               "will fallback to use OpenCV instead."
            << std::endl;
#endif
}

void DisableFlyCV() {
  Processor::default_lib = ProcLib::OPENCV;
  FDINFO << "Will change to use image processing library "
         << Processor::default_lib << std::endl;
}

cv::Mat CreateOpenCVMatFromTensor(const FDTensor& tensor) {
  FDDataType type = tensor.dtype;
  FDASSERT(tensor.shape.size() == 3,
           "When create FD Mat from tensor, tensor shape should be 3-Dim, HWC "
           "layout");
  int64_t height = tensor.shape[0];
  int64_t width = tensor.shape[1];
  int64_t channel = tensor.shape[2];
  cv::Mat ocv_mat;
  // reference to outside FDTensor, zero copy
  switch (type) {
    case FDDataType::UINT8:
      ocv_mat = cv::Mat(height, width, CV_8UC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::INT8:
      ocv_mat = cv::Mat(height, width, CV_8SC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::INT16:
      ocv_mat = cv::Mat(height, width, CV_16SC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::INT32:
      ocv_mat = cv::Mat(height, width, CV_32SC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::FP32:
      ocv_mat = cv::Mat(height, width, CV_32FC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::FP64:
      ocv_mat = cv::Mat(height, width, CV_64FC(channel),
                        const_cast<void*>(tensor.Data()));
      break;
    default:
      FDASSERT(false,
               "Tensor type %d is not supported While calling "
               "CreateFDMatFromTensor.",
               type);
      break;
  }
  return ocv_mat;
}

#ifdef ENABLE_FLYCV
fcv::Mat CreateFlyCVMatFromTensor(const FDTensor& tensor) {
  FDDataType type = tensor.dtype;
  FDASSERT(tensor.shape.size() == 3,
           "When create FD Mat from tensor, tensor shape should be 3-Dim, HWC "
           "layout");
  int64_t height = tensor.shape[0];
  int64_t width = tensor.shape[1];
  int64_t channel = tensor.shape[2];
  fcv::Mat fcv_mat;
  auto fcv_type = CreateFlyCVDataType(type, static_cast<int>(channel));
  switch (type) {
    case FDDataType::UINT8:
      fcv_mat =
        fcv::Mat(width, height, fcv_type, const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::FP32:
      fcv_mat =
        fcv::Mat(width, height, fcv_type, const_cast<void*>(tensor.Data()));
      break;
    case FDDataType::FP64:
      fcv_mat =
        fcv::Mat(width, height, fcv_type, const_cast<void*>(tensor.Data()));
    break;
    default:
      FDASSERT(false,
              "Tensor type %d is not supported While calling "
              "CreateFDMatFromTensor.",
               type);
    break;
  }
  return fcv_mat;
}
#endif

Mat CreateFDMatFromTensor(const FDTensor& tensor) {
  if (Processor::default_lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat fcv_mat = CreateFlyCVMatFromTensor(tensor);
    Mat mat = Mat(fcv_mat);
    return mat;
#else
    FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
  }
  cv::Mat ocv_mat = CreateOpenCVMatFromTensor(tensor);
  Mat mat = Mat(ocv_mat);
  return mat;
}

}  // namespace vision
}  // namespace fastdeploy
