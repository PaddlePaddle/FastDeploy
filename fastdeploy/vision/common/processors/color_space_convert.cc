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
bool BGR2RGB::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool BGR2RGB::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::cuda::GpuMat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
  mat->SetMat(new_im);
  return true;
}
#endif

bool RGB2BGR::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool RGB2BGR::GpuRun(Mat* mat) {
  cv::Mat new_im;
  cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
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
