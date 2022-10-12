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

#include "fastdeploy/vision/common/processors/crop.h"

namespace fastdeploy {
namespace vision {

bool Crop::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ + offset_h_ || width < width_ + offset_w_) {
    FDERROR << "[Crop] Cannot crop [" << height_ << ", " << width_
            << "] from the input image [" << height << ", " << width
            << "], with offset [" << offset_h_ << ", " << offset_w_ << "]."
            << std::endl;
    return false;
  }
  cv::Rect crop_roi(offset_w_, offset_h_, width_, height_);
  *im = (*im)(crop_roi);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}

bool Crop::ImplByOpenCV(const Mat& src_im, Mat* dst_im) {
  const cv::Mat* img = src_im.GetCpuMat();
  cv::Mat* crop_img = dst_im->GetCpuMat();
  int crop_x1 = std::max(0, area_[0]);
  int crop_y1 = std::max(0, area_[1]);
  int crop_x2 = std::min(img->cols - 1, area_[2]);
  int crop_y2 = std::min(img->rows - 1, area_[3]);
  int center_x = (crop_x1 + crop_x2) / 2.;
  int center_y = (crop_y1 + crop_y2) / 2.;
  int half_h = (crop_y2 - crop_y1) / 2.;
  int half_w = (crop_x2 - crop_x1) / 2.;
  // adjust h or w to keep image ratio, expand the shorter edge
  if (half_h * 3 > half_w * 4) {
    half_w = static_cast<int>(half_h * 0.75);
  } else {
    half_h = static_cast<int>(half_w * 4 / 3);
  }

  crop_x1 =
      std::max(0, center_x - static_cast<int>(half_w * (1 + expandratio_)));
  crop_y1 =
      std::max(0, center_y - static_cast<int>(half_h * (1 + expandratio_)));
  crop_x2 = std::min(img->cols - 1,
                     static_cast<int>(center_x + half_w * (1 + expandratio_)));
  crop_y2 = std::min(img->rows - 1,
                     static_cast<int>(center_y + half_h * (1 + expandratio_)));
  *crop_img =
      (*img)(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
  center_->clear();
  center_->emplace_back((crop_x1 + crop_x2) / 2);
  center_->emplace_back((crop_y1 + crop_y2) / 2);

  scale_->clear();
  scale_->emplace_back((crop_x2 - crop_x1));
  scale_->emplace_back((crop_y2 - crop_y1));
  return true;
}

bool Crop::Run(Mat* mat, int offset_w, int offset_h, int width, int height,
               ProcLib lib) {
  auto c = Crop(offset_w, offset_h, width, height);
  return c(mat, lib);
}

bool Crop::Run(const Mat& src_im, Mat* dst_im,
                const std::vector<int>& area, std::vector<float>* center,
                std::vector<float>* scale, const float expandratio,
                ProcLib lib) {
  auto c = Crop(area, center, scale, expandratio);
  return c(src_im, dst_im, lib);
}

}  // namespace vision
}  // namespace fastdeploy
