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

#include "fastdeploy/vision/common/processors/color_space_convert.h"

namespace fastdeploy {
namespace vision {
bool BGR2RGB::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_BGR2RGB);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool BGR2RGB::ImplByOpenCVCuda(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetOpenCVCudaMat();
  cv::cuda::GpuMat new_im;
  cv::cuda::createContinuous(im->rows, im->cols, im->type(), new_im);
  cv::cuda::cvtColor(*im, new_im, cv::COLOR_BGR2RGB);
  mat->SetMat(new_im);
  FDINFO << new_im.isContinuous() << std::endl;
  return true;
}
#endif

#ifdef ENABLE_FLYCV
bool BGR2RGB::ImplByFalconCV(Mat* mat) {
  fcv::Mat* im = mat->GetFalconCVMat();
  if (im->channels() != 3) {
    FDERROR << "[BGR2RGB] The channel of input image must be 3, but not it's " << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_BGR2PA_RGB);
  mat->SetMat(new_im);
  return true;
}
#endif

bool RGB2BGR::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_FLYCV
bool RGB2BGR::ImplByFalconCV(Mat* mat) {
  fcv::Mat* im = mat->GetFalconCVMat();
  if (im->channels() != 3) {
    FDERROR << "[RGB2BGR] The channel of input image must be 3, but not it's " << im->channels() << "." << std::endl;
    return false;
  }
  fcv::Mat new_im;
  fcv::cvt_color(*im, new_im, fcv::ColorConvertType::CVT_PA_RGB2PA_BGR);
  mat->SetMat(new_im);
  return true;
}
#endif

bool BGR2RGB::Run(Mat* mat, ProcLib lib) {
  auto b = BGR2RGB();
  return b(mat, lib);
}

bool RGB2BGR::Run(Mat* mat, ProcLib lib) {
  auto r = RGB2BGR();
  return r(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
