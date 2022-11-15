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
#include "fastdeploy/vision/common/processors/proc_lib.h"

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace vision {

bool Processor::operator()(Mat* mat, ProcLib lib) {
  ProcLib target = lib;
  if (lib == ProcLib::DEFAULT) {
    target = DefaultProcLib::default_lib;
  }
  if (target == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return ImplByFlyCV(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with FlyCV.");
#endif
  } else if (target == ProcLib::CUDA) {
#ifdef WITH_GPU
    return ImplByCuda(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif
  }
  // DEFAULT & OPENCV
  return ImplByOpenCV(mat);
}

FDTensor* Processor::UpdateAndGetReusedBuffer(
    const std::vector<int64_t>& new_shape, const int& opencv_dtype,
    const std::string& buffer_name, const Device& new_device,
    const bool& use_pinned_memory) {
  if (reused_buffers_.count(buffer_name) == 0) {
    reused_buffers_[buffer_name] = FDTensor();
  }
  reused_buffers_[buffer_name].is_pinned_memory = use_pinned_memory;
  reused_buffers_[buffer_name].Resize(new_shape,
                                      OpenCVDataTypeToFD(opencv_dtype),
                                      buffer_name, new_device);
  return &reused_buffers_[buffer_name];
}

void EnableFlyCV() {
#ifdef ENABLE_FLYCV
  DefaultProcLib::default_lib = ProcLib::FLYCV;
  FDINFO << "Will change to use image processing library "
         << DefaultProcLib::default_lib << std::endl;
#else
  FDWARNING << "FastDeploy didn't compile with FlyCV, "
               "will fallback to use OpenCV instead."
            << std::endl;
#endif
}

void DisableFlyCV() {
  DefaultProcLib::default_lib = ProcLib::OPENCV;
  FDINFO << "Will change to use image processing library "
         << DefaultProcLib::default_lib << std::endl;
}

void SetProcLibCpuNumThreads(int threads) {
  cv::setNumThreads(threads);
#ifdef ENABLE_FLYCV
  fcv::set_thread_num(threads);
#endif
}

}  // namespace vision
}  // namespace fastdeploy
