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

#ifdef ENABLE_FLYCV
bool LimitLong::ImplByFalconCV(Mat* mat) {
  fcv::Mat* im = mat->GetFalconCVMat();
  int origin_w = im->width();
  int origin_h = im->height();
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
    if (fabs(scale - 1.0) < 1e-06) {
      return true;
    }
    auto interp_method = fcv::InterpolationType::INTER_LINEAR;
    if (interp_ == 0) {
      interp_method = fcv::InterpolationType::INTER_NEAREST;
    } else if (interp_ == 1) {
      interp_method = fcv::InterpolationType::INTER_LINEAR;
    } else if (interp_ == 2) {
      interp_method = fcv::InterpolationType::INTER_CUBIC;
    } else {
      FDERROR << "LimitLong: Only support interp_ be 0/1/2 with FalconCV, but "
                 "now it's "
              << interp_ << "." << std::endl;
      return false;
    }
    fcv::Mat new_im;
    FDERROR << "origin " << im->width() << " " << im->height() << std::endl;
    FDERROR << "scale " << scale << std::endl;
    fcv::resize(*im, new_im, fcv::Size(), scale, scale, interp_method);
    FDERROR << "after " << new_im.width() << " " << new_im.height() << std::endl;
    mat->SetMat(new_im);
    mat->SetWidth(new_im.width());
    mat->SetHeight(new_im.height());
  }
  return true;
}
#endif

bool LimitLong::Run(Mat* mat, int max_long, int min_long, int interp, ProcLib lib) {
  auto l = LimitLong(max_long, min_long, interp);
  return l(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
