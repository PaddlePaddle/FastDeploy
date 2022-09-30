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

#include "fastdeploy/vision/common/processors/resize_by_short.h"

namespace fastdeploy {
namespace vision {

bool ResizeByShort::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  double scale = GenerateScale(origin_w, origin_h);
  if (use_scale_ && fabs(scale - 1.0) >= 1e-06) {
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
  } else {
    int width = static_cast<int>(round(scale * im->cols));
    int height = static_cast<int>(round(scale * im->rows));
    if (width != origin_w || height != origin_h) {
      cv::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
    }
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool ResizeByShort::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  double scale = GenerateScale(origin_w, origin_h);
  im->convertTo(*im, CV_32FC(im->channels()));
  if (use_scale_ && fabs(scale - 1.0) >= 1e-06) {
    cv::cuda::resize(*im, *im, cv::Size(), scale, scale, interp_);
  } else {
    int width = static_cast<int>(round(scale * im->cols));
    int height = static_cast<int>(round(scale * im->rows));
    if (width != origin_w || height != origin_h) {
      cv::cuda::resize(*im, *im, cv::Size(width, height), 0, 0, interp_);
    }
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}
#endif

double ResizeByShort::GenerateScale(const int origin_w, const int origin_h) {
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  double scale =
      static_cast<double>(target_size_) / static_cast<double>(im_size_min);
  if (max_hw_.size() > 0) {
    FDASSERT(max_hw_.size() == 2,
             "Require size of max_hw_ be 2, but now it's %zu.", max_hw_.size());
    FDASSERT(
        max_hw_[0] > 0 && max_hw_[1] > 0,
        "Require elements in max_hw_ greater than 0, but now it's [%d, %d].",
        max_hw_[0], max_hw_[1]);
    if (round(scale * origin_h) > max_hw_[0]) {
      scale = static_cast<double>(max_hw_[0]) / static_cast<double>(origin_h);
    }
    if (round(scale * origin_w) > max_hw_[1]) {
      scale = static_cast<double>(max_hw_[1]) / static_cast<double>(origin_w);
    }
  }
  return scale;
}

bool ResizeByShort::Run(Mat* mat, int target_size, int interp, bool use_scale,
                        const std::vector<int>& max_hw, ProcLib lib) {
  auto r = ResizeByShort(target_size, interp, use_scale, max_hw);
  return r(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
