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

class ResizeByInputShape : public Processor {
 public:
  ResizeByInputShape(int width, int height, int interp = 1) {
    width_ = width;
    height_ = height;
    interp_ = interp;
  }
  // Resize input Mat* mat(origin_w, origin_h) by the input_shape (width_,
  // height_).
  // If any edge of mat is larger than that of input_shape, ResizeByInputShape
  // will compute the smallest scale(s) of {width_/origin_w, height_/origin_h}
  // and resize mat by s.
  bool CpuRun(Mat* mat);
#ifdef ENABLE_OPENCV_CUDA
  bool GpuRun(Mat* mat);
#endif
  std::string Name() { return "ResizeByInputShape"; }

  static bool Run(Mat* mat, int width, int height, int interp = 1,
                  ProcLib lib = ProcLib::OPENCV_CPU);

 private:
  int width_;
  int height_;
  int interp_ = 1;
};
}  // namespace vision
}  // namespace fastdeploy
