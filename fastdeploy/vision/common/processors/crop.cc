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
  cv::Mat* im = mat->GetOpenCVMat();
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
  const cv::Mat* img = src_im.GetOpenCVMat();
  cv::Mat* crop_img = dst_im->GetOpenCVMat();
  int xmin = static_cast<int>(area_[0]);
  int ymin = static_cast<int>(area_[1]);
  int xmax = static_cast<int>(area_[2]);
  int ymax = static_cast<int>(area_[3]);
  float center_x = (xmin + xmax) / 2.0f;
  float center_y = (ymin + ymax) / 2.0f;
  float half_h = (ymax - ymin) * (1 + expandratio_) / 2.0f;
  float half_w = (xmax - xmin) * (1 + expandratio_) / 2.0f;
  // adjust h or w to keep image ratio, expand the shorter edge
  if (half_h * 3 > half_w * 4) {
    half_w = half_h * 0.75;
  } 
  int crop_xmin =std::max(0, static_cast<int>(center_x - half_w));
  int crop_ymin =std::max(0, static_cast<int>(center_y - half_h));
  int crop_xmax = std::min(img->cols - 1, static_cast<int>(center_x + half_w));
  int crop_ymax = std::min(img->rows - 1, static_cast<int>(center_y + half_h));
  
  crop_img->create(crop_ymax - crop_ymin, crop_xmax - crop_xmin, img->type());
  *crop_img =
      (*img)(cv::Range(crop_ymin, crop_ymax), cv::Range(crop_xmin, crop_xmax));
  center_->clear();
  center_->emplace_back((crop_xmin + crop_xmax) / 2.0f);
  center_->emplace_back((crop_ymin + crop_ymax) / 2.0f);

  scale_->clear();
  scale_->emplace_back((crop_xmax - crop_xmin));
  scale_->emplace_back((crop_ymax - crop_ymin));

  dst_im->SetWidth(crop_img->cols);
  dst_im->SetHeight(crop_img->rows);
  return true;
}

bool Crop::Run(Mat* mat, int offset_w, int offset_h, int width, int height,
               ProcLib lib) {
  auto c = Crop(offset_w, offset_h, width, height);
  return c(mat, lib);
}

bool Crop::Run(const Mat& src_im, Mat* dst_im,
                const std::vector<float>& area, std::vector<float>* center,
                std::vector<float>* scale, const float expandratio,
                ProcLib lib) {
  auto c = Crop(area, center, scale, expandratio);
  return c(src_im, dst_im, lib);
}

}  // namespace vision
}  // namespace fastdeploy
