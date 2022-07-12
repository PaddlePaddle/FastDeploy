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
  cv::Mat* im = mat->GetCpuMat();
  int c = im->channels();
  if (dtype_ == "float") {
    if (im->type() != CV_32FC(c)) {
      im->convertTo(*im, CV_32FC(c));
    }
  } else if (dtype_ == "double") {
    if (im->type() != CV_64FC(c)) {
      im->convertTo(*im, CV_64FC(c));
    }
  } else {
    FDLogger() << "[WARN] Cast not support for " << dtype_ 
               << " now! will skip this operation."
               << std::endl;
  }
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Cast::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int c = im->channels();
  if (dtype_ == "float") {
    if (im->type() != CV_32FC(c)) {
      im->convertTo(*im, CV_32FC(c));
    }
  } else if (dtype_ == "double") {
    if (im->type() != CV_64FC(c)) {
      im->convertTo(*im, CV_64FC(c));
    }
  } else {
    FDLogger() << "[WARN] Cast not support for " << dtype_ 
               << " now! will skip this operation."
               << std::endl;
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
