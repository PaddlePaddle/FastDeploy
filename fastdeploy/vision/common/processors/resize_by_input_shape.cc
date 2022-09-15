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

#include "fastdeploy/vision/common/processors/resize_by_input_shape.h"

namespace fastdeploy {
namespace vision {

bool ResizeByInputShape::CpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "ResizeByInputShape: The format of input is not HWC."
            << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  float scale_w = width_ * 1.0 / origin_w;
  float scale_h = height_ * 1.0 / origin_h;
  float scale = std::min(scale_w, scale_h);
  cv::resize(*im, *im, cv::Size(0, 0), scale, scale, interp_);
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool ResizeByInputShape::GpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "ResizeByInputShape: The format of input is not HWC."
            << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  float scale_w = width_ * 1.0 / origin_w;
  float scale_h = height_ * 1.0 / origin_h;
  float scale = std::min(scale_w, scale_h);
  cv::cuda::resize(*im, *im, cv::Size(0, 0), scale, scale, interp_);
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}
#endif

bool ResizeByInputShape::Run(Mat* mat, int width, int height, int interp,
                             ProcLib lib) {
  if (mat->Height() == height && mat->Width() == width) {
    return true;
  }
  auto r = ResizeByInputShape(width, height, interp);
  return r(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
