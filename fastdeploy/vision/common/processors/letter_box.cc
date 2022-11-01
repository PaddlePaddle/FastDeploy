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

#include "fastdeploy/vision/common/processors/letter_box.h"

namespace fastdeploy{
namespace vision{
bool LetterBoxResize::ImplByOpenCV(Mat* mat) {

  if (mat->Channels() != color_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << color_.size() << "."
            << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetOpenCVMat();
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  // get_resized_shape
  int new_shape_w = std::round(im->cols * resize_scale);
  int new_shape_h = std::round(im->rows * resize_scale);
  // calculate pad
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);
  cv::resize(*im, *im, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);
  cv::Scalar color;
  if (color_.size() == 1) {
      color = cv::Scalar(color_[0]);
  } else if (color_.size() == 2) {
      color = cv::Scalar(color_[0], color_[1]);
  } else if (color_.size() == 3) {
      color = cv::Scalar(color_[0], color_[1], color_[2]);
  } else {
      color = cv::Scalar(color_[0], color_[1], color_[2], color_[3]);
  }
  cv::copyMakeBorder(*im, *im, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

bool LetterBoxResize::Run(Mat* mat, const std::vector<int>& target_size, const std::vector<float>& color, ProcLib lib) {
    auto l = LetterBoxResize(target_size,color);
    return l(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
