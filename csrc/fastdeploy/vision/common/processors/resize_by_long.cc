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

#include "fastdeploy/vision/common/processors/resize_by_long.h"

namespace fastdeploy {
namespace vision {

bool ResizeByLong::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  double scale = GenerateScale(origin_w, origin_h);
  if (use_scale_) {
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
  } else {
    int width = static_cast<int>(round(scale * im->cols));
    int height = static_cast<int>(round(scale * im->rows));
    cv::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool ResizeByLong::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  double scale = GenerateScale(origin_w, origin_h);
  im->convertTo(*im, CV_32FC(im->channels()));
  if (use_scale_) {
    cv::cuda::resize(*im, *im, cv::Size(), scale, scale, interp_);
  } else {
    int width = static_cast<int>(round(scale * im->cols));
    int height = static_cast<int>(round(scale * im->rows));
    cv::cuda::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}
#endif

double ResizeByLong::GenerateScale(const int origin_w, const int origin_h) {
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  double scale = 1.0f;
  if (target_size_ == -1) {
    if (im_size_max > max_size_) {
      scale = static_cast<double>(max_size_) / static_cast<double>(im_size_max);
    }
  } else {
    scale =
        static_cast<double>(target_size_) / static_cast<double>(im_size_max);
  }
  return scale;
}

bool ResizeByLong::Run(Mat* mat, int target_size, int interp, bool use_scale,
                       int max_size, ProcLib lib) {
  auto r = ResizeByLong(target_size, interp, use_scale, max_size);
  return r(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
