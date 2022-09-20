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

#include "fastdeploy/vision/common/processors/limit_short.h"

namespace fastdeploy {
namespace vision {

bool LimitShort::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  int im_size_min = std::min(origin_w, origin_h);
  int target = im_size_min;
  if (max_short_ > 0 && im_size_min > max_short_) {
    target = max_short_;
  } else if (min_short_ > 0 && im_size_min < min_short_) {
    target = min_short_;
  }
  double scale = -1.f;
  if (target != im_size_min) {
    scale = static_cast<double>(target) / static_cast<double>(im_size_min);
  }
  // 如果给了固定的shape，按照固定shape就好。
  if (input_w_ > 0 && input_h_ > 0) {
    // 给出的input_shape 大于原始的shape(origin_w, origin_h) 则直接返回。
    if (origin_w <= input_w_ && origin_h <= input_h_) {
      return true;
    }
    float scale_w = input_w_ * 1.0 / origin_w;
    float scale_h = input_h_ * 1.0 / origin_h;
    scale = std::min(scale_w, scale_h);
  }
  if (scale > 0) {
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool LimitShort::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  im->convertTo(*im, CV_32FC(im->channels()));
  int im_size_min = std::min(origin_w, origin_h);
  int target = im_size_min;
  if (max_short_ > 0 && im_size_min > max_short_) {
    target = max_short_;
  } else if (min_short_ > 0 && im_size_min < min_short_) {
    target = min_short_;
  }
  double scale = -1.f;
  if (target != im_size_min) {
    scale = static_cast<double>(target) / static_cast<double>(im_size_min);
  }
  // 如果给了固定的shape，按照固定shape就好。
  if (input_w_ > 0 && input_h_ > 0) {
    // 给出的input_shape 大于原始的shape(origin_w, origin_h) 则直接返回。
    if (origin_w <= input_w_ && origin_h <= input_h_) {
      return true;
    }
    float scale_w = input_w_ * 1.0 / origin_w;
    float scale_h = input_h_ * 1.0 / origin_h;
    scale = std::min(scale_w, scale_h);
  }
  if (scale > 0) {
    cv::cuda::resize(*im, *im, cv::Size(), scale, scale, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}
#endif

bool LimitShort::Run(Mat* mat, int max_short, int min_short, int input_w,
                     int input_h, ProcLib lib) {
  auto l = LimitShort(max_short, min_short, input_w, input_h);
  return l(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
