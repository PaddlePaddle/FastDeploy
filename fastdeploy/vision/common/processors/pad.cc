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

#include "fastdeploy/vision/common/processors/pad.h"

namespace fastdeploy {
namespace vision {

bool Pad::CpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Pad: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "Pad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << value_.size() << "."
            << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  cv::Mat new_im;
  cv::copyMakeBorder(*im, new_im, top_, bottom_, left_, right_,
                     cv::BORDER_CONSTANT, value);
  mat->SetMat(new_im);
  mat->SetHeight(im->rows);
  mat->SetWidth(im->cols);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Pad::GpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Pad: The input data must be Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "Pad: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR << "Pad: Require input channels equals to size of padding value, "
               "but now channels = "
            << mat->Channels()
            << ", the size of padding values = " << value_.size() << "."
            << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  cv::cuda::copyMakeBorder(*im, *im, top_, bottom_, left_, right_,
                           cv::BORDER_CONSTANT, value);
  mat->SetHeight(im->rows);
  mat->SetWidth(im->cols);
  return true;
}
#endif

bool Pad::Run(Mat* mat, const int& top, const int& bottom, const int& left,
              const int& right, const std::vector<float>& value, ProcLib lib) {
  auto p = Pad(top, bottom, left, right, value);
  return p(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
