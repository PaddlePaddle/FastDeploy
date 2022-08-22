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

#include <map>

#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

void cls_resize_img(const cv::Mat &img, cv::Mat &resize_img,
                    const std::vector<int> &rec_image_shape) {
  int imgC = rec_image_shape[0];
  int imgH = rec_image_shape[1];
  int imgW = rec_image_shape[2];

  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  if (resize_w < imgW) {
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  }
}

void crnn_resize_img(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                     const std::vector<int> &rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(imgH * wh_ratio);

  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;

  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                     int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                     {127, 127, 127});
}
}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy