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

#pragma once

#include "fastdeploy/vision/common/processors/base.h"

namespace fastdeploy {
namespace vision {

class Crop : public Processor {
 public:
  Crop(int offset_w, int offset_h, int width, int height) {
    offset_w_ = offset_w;
    offset_h_ = offset_h;
    width_ = width;
    height_ = height;
  }

  Crop(const std::vector<float> area,
       std::vector<float>* center,
       std::vector<float>* scale,
       const float expandratio) {
    area_ = area;
    center_ = center;
    scale_ = scale;
    expandratio_ = expandratio;
  }
  bool ImplByOpenCV(Mat* mat);
  bool ImplByOpenCV(const Mat& src_im, Mat* dst_im);
  std::string Name() { return "Crop"; }

  static bool Run(Mat* mat, int offset_w, int offset_h, int width, int height,
                  ProcLib lib = ProcLib::OPENCV);
  static bool Run(const Mat& src_im, Mat* dst_im,
                  const std::vector<float>& area, std::vector<float>* center,
                  std::vector<float>* scale, const float expandratio = 0.3,
                  ProcLib lib = ProcLib::OPENCV);

 private:
  int offset_w_;
  int offset_h_;
  int height_;
  int width_;

  std::vector<float> area_;
  std::vector<float>* center_;
  std::vector<float>* scale_;
  float expandratio_;
};

}  // namespace vision
}  // namespace fastdeploy
