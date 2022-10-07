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

#include "fastdeploy/vision/common/processors/limit_by_stride.h"

namespace fastdeploy {
namespace vision {

bool LimitByStride::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  int rw = origin_w - origin_w % stride_;
  int rh = origin_h - origin_h % stride_;
  if (rw != origin_w || rh != origin_w) {
    cv::resize(*im, *im, cv::Size(rw, rh), 0, 0, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool LimitByStride::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  im->convertTo(*im, CV_32FC(im->channels()));
  int rw = origin_w - origin_w % stride_;
  int rh = origin_h - origin_h % stride_;
  if (rw != origin_w || rh != origin_w) {
    cv::cuda::resize(*im, *im, cv::Size(rw, rh), 0, 0, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}
#endif

bool LimitByStride::Run(Mat* mat, int stride, int interp, ProcLib lib) {
  auto r = LimitByStride(stride, interp);
  return r(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
