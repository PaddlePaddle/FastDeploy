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

#include "fastdeploy/vision/common/processors/center_crop.h"

namespace fastdeploy {
namespace vision {

bool CenterCrop::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cv::Rect crop_roi(offset_x, offset_y, width_, height_);
  *im = (*im)(crop_roi);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}

bool CenterCrop::Run(Mat* mat, const int& width, const int& height,
                     ProcLib lib) {
  auto c = CenterCrop(width, height);
  return c(mat, lib);
}

} // namespace vision
} // namespace fastdeploy
