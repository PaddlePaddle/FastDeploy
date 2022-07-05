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

#include "fastdeploy/vision/common/processors/cast.h"

namespace fastdeploy {
namespace vision {

bool Cast::CpuRun(Mat* mat) {
  if (mat->layout != Layout::CHW) {
    FDERROR << "Cast: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  if (dtype_ == "float") {
    im->convertTo(*im, CV_32FC(im->channels()));
  } else if (dtype_ == "double") {
    im->convertTo(*im, CV_64FC(im->channels()));
  }
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Cast::GpuRun(Mat* mat) {
  if (mat->layout != Layout::CHW) {
    FDERROR << "Cast: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  if (dtype_ == "float") {
    im->convertTo(*im, CV_32FC(im->channels()));
  } else if (dtype_ == "double") {
    im->convertTo(*im, CV_64FC(im->channels()));
  }
  return true;
}
#endif

bool Cast::Run(Mat* mat, const std::string& dtype, ProcLib lib) {
  auto c = Cast(dtype);
  return c(mat, lib);
}

} // namespace vision
} // namespace fastdeploy
