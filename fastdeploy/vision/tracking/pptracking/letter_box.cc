//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision/tracking/pptracking/letter_box.h"

namespace fastdeploy {
namespace vision {
namespace tracking {

LetterBoxResize::LetterBoxResize(const std::vector<int>& target_size, const std::vector<float>& color){
  target_size_=target_size;
  color_=color;
}
bool LetterBoxResize::ImplByOpenCV(Mat* mat){
  if (mat->Channels() != color_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
    "but now channels = "
    << mat->Channels()
    << ", the size of padding values = " << color_.size() << "."
    << std::endl;
    return false;
  }
  // generate scale_factor
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  int target_h = target_size_[0];
  int target_w = target_size_[1];
  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);

  int new_shape_w = std::round(mat->Width() * resize_scale);
  int new_shape_h = std::round(mat->Height() * resize_scale);
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;
  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);

  Resize::Run(mat,new_shape_w,new_shape_h);
  Pad::Run(mat,top,bottom,left,right,color_);
  return true;
}

} // namespace tracking
} // namespace vision
} // namespace fastdeploy