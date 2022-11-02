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

#include "fastdeploy/vision/common/processors/limit_long.h"

namespace fastdeploy {
namespace vision {

bool LimitLong::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  int im_size_max = std::max(origin_w, origin_h);
  int target = im_size_max;
  if (max_long_ > 0 && im_size_max > max_long_) {
    target = max_long_;
  } else if (min_long_ > 0 && im_size_max < min_long_) {
    target = min_long_;
  }
  if (target != im_size_max) {
    double scale =
        static_cast<double>(target) / static_cast<double>(im_size_max);
    cv::resize(*im, *im, cv::Size(), scale, scale, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}

bool LimitLong::Run(Mat* mat, int max_long, int min_long, int interp, ProcLib lib) {
  auto l = LimitLong(max_long, min_long, interp);
  return l(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
