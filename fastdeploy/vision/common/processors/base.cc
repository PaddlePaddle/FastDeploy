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

#include "fastdeploy/vision/common/processors/base.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace vision {

ProcLib Processor::default_lib = ProcLib::DEFAULT;

bool Processor::CpuRun(Mat* mat) {
  FDERROR << "Unimplemented CpuRun." << std::endl;
  return false;
}

#ifdef ENABLE_OPENCV_CUDA
bool Processor::GpuRun(Mat* mat) {
  FDERROR << "Unimplemented GpuRun." << std::endl;
  return false;
}
#endif

bool Processor::operator()(Mat* mat, ProcLib lib) {
  // if default_lib is set
  // then use default_lib
  ProcLib target = lib;
  if (default_lib != ProcLib::DEFAULT) {
    target = default_lib;
  }

  if (target == ProcLib::OPENCV_CUDA) {
#ifdef ENABLE_OPENCV_CUDA
    bool ret = GpuRun(mat);
    mat->device = Device::GPU;
    return ret;
#else
    FDERROR
        << "OpenCV is not compiled with CUDA, cannot process image with CUDA."
        << std::endl;
    return false;
#endif
  }
  bool ret = CpuRun(mat);
  mat->device = Device::CPU;
  return ret;
}

} // namespace vision
} // namespace fastdeploy
