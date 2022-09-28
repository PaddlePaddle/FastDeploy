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

#include "fastdeploy/utils/utils.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace fastdeploy {
namespace vision {

enum ProcLib { DEFAULT, OPENCV_CPU, OPENCV_CUDA };

class Processor {
 public:
  // default_lib has the highest priority
  // all the function in `processor` will force to use
  // default_lib if this flag is set.
  // DEFAULT means this flag is not set
  static ProcLib default_lib;

  //  virtual bool ShapeInfer(const std::vector<int>& in_shape,
  //                          std::vector<int>* out_shape) = 0;
  virtual std::string Name() = 0;
  virtual bool CpuRun(Mat* mat);
#ifdef ENABLE_OPENCV_CUDA
  virtual bool GpuRun(Mat* mat);
#endif

  virtual bool operator()(Mat* mat,
                          ProcLib lib = ProcLib::OPENCV_CPU);
};

} // namespace vision
} // namespace fastdeploy
