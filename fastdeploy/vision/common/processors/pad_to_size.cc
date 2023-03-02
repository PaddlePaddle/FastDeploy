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

#include "fastdeploy/vision/common/processors/pad_to_size.h"

namespace fastdeploy {
namespace vision {

static bool PadHWCByOpenCV(FDMat* mat, int width, int height,
                           const std::vector<float>& value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  if (origin_w > width) {
    FDERROR << "PadToSize: the input width:" << origin_w
            << " is greater than the target width: " << width << "."
            << std::endl;
    return false;
  }
  if (origin_h > height) {
    FDERROR << "PadToSize: the input height:" << origin_h
            << " is greater than the target height: " << height << "."
            << std::endl;
    return false;
  }
  if (origin_w == width && origin_h == height) {
    return true;
  }

  cv::Mat* im = mat->GetOpenCVMat();
  cv::Scalar scalar;
  if (value.size() == 1) {
    scalar = cv::Scalar(value[0]);
  } else if (value.size() == 2) {
    scalar = cv::Scalar(value[0], value[1]);
  } else if (value.size() == 3) {
    scalar = cv::Scalar(value[0], value[1], value[2]);
  } else {
    scalar = cv::Scalar(value[0], value[1], value[2], value[3]);
  }
  // top, bottom, left, right
  cv::copyMakeBorder(*im, *im, 0, height - origin_h, 0, width - origin_w,
                     cv::BORDER_CONSTANT, scalar);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

static bool PadCHWByOpenCV(FDMat* mat, int width, int height,
                           const std::vector<float>& value) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  cv::Mat new_im(height, width,
                 CreateOpenCVDataType(mat->Type(), mat->Channels()));
  for (int i = 0; i < mat->Channels(); ++i) {
    uint8_t* src_data = reinterpret_cast<uint8_t*>(mat->Data()) +
                        i * origin_w * origin_h * FDDataTypeSize(mat->Type());
    cv::Mat src(origin_h, origin_w, CreateOpenCVDataType(mat->Type(), 1),
                src_data);

    uint8_t* dst_data =
        new_im.ptr() + i * width * height * FDDataTypeSize(mat->Type());
    cv::Mat dst(height, width, CreateOpenCVDataType(mat->Type(), 1), dst_data);

    cv::copyMakeBorder(src, dst, 0, height - origin_h, 0, width - origin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value[i]));
  }
  mat->SetMat(new_im);
  mat->SetHeight(height);
  mat->SetWidth(width);
  return true;
}

bool PadToSize::ImplByOpenCV(FDMat* mat) {
  if (width_ == -1 || height_ == -1) {
    return true;
  }
  if (mat->Channels() > 4) {
    FDERROR << "PadToSize: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "PadToSize: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  if (origin_w > width_) {
    FDERROR << "PadToSize: the input width:" << origin_w
            << " is greater than the target width: " << width_ << "."
            << std::endl;
    return false;
  }
  if (origin_h > height_) {
    FDERROR << "PadToSize: the input height:" << origin_h
            << " is greater than the target height: " << height_ << "."
            << std::endl;
    return false;
  }
  if (origin_w == width_ && origin_h == height_) {
    return true;
  }
  if (mat->layout == Layout::HWC) {
    return PadHWCByOpenCV(mat, width_, height_, value_);
  } else if (mat->layout == Layout::CHW) {
    return PadCHWByOpenCV(mat, width_, height_, value_);
  }
  return false;
}

#ifdef ENABLE_FLYCV
bool PadToSize::ImplByFlyCV(Mat* mat) {
  if (width_ == -1 || height_ == -1) {
    return true;
  }
  if (mat->layout != Layout::HWC) {
    FDERROR << "PadToSize: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "PadToSize: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "PadToSize: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  if (origin_w > width_) {
    FDERROR << "PadToSize: the input width:" << origin_w
            << " is greater than the target width: " << width_ << "."
            << std::endl;
    return false;
  }
  if (origin_h > height_) {
    FDERROR << "PadToSize: the input height:" << origin_h
            << " is greater than the target height: " << height_ << "."
            << std::endl;
    return false;
  }
  if (origin_w == width_ && origin_h == height_) {
    return true;
  }

  fcv::Mat* im = mat->GetFlyCVMat();
  fcv::Scalar value;
  if (value_.size() == 1) {
    value = fcv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = fcv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = fcv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = fcv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  fcv::Mat new_im;
  // top, bottom, left, right
  fcv::copy_make_border(*im, new_im, 0, height_ - origin_h, 0,
                        width_ - origin_w, fcv::BorderType::BORDER_CONSTANT,
                        value);
  mat->SetMat(new_im);
  mat->SetHeight(height_);
  mat->SetWidth(width_);
  return true;
}
#endif

bool PadToSize::Run(Mat* mat, int width, int height,
                    const std::vector<float>& value, ProcLib lib) {
  auto p = PadToSize(width, height, value);
  return p(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
