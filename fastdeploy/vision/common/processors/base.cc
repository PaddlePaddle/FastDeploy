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
  // if default_lib is set
  // then use default_lib
  ProcLib target = lib;
  if (default_lib != ProcLib::DEFAULT) {
    target = default_lib;
  }

  if (target == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    if (mat->mat_type != ProcLib::FLYCV) {
      if (mat->layout != Layout::HWC) {
        FDERROR << "Cannot convert cv::Mat to fcv::Mat while layout is not HWC." << std::endl;
      }
      fcv::Mat fcv_mat = ConvertOpenCVMatToFalconCV(*(mat->GetOpenCVMat()));
      mat->SetMat(fcv_mat);
    }
    return ImplByFalconCV(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with FalconCV.");
#endif
  } else if (target == ProcLib::OPENCVCUDA) {
#ifdef ENABLE_OPENCV_CUDA
    if (mat->mat_type != ProcLib::OPENCVCUDA) {
      cv::cuda::GpuMat gpu_mat;
      cv::cuda::createContinuous(mat->GetOpenCVMat()->rows, mat->GetOpenCVMat()->cols,
                                 mat->GetOpenCVMat()->type(), gpu_mat);
      gpu_mat.upload(*(mat->GetOpenCVMat()));
      mat->SetMat(gpu_mat);
    }
    return ImplByOpenCVCuda(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with OpenCV_CUDA.");
#endif
  }
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

}  // namespace vision
}  // namespace fastdeploy
