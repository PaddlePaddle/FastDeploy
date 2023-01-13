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

namespace fastdeploy {
namespace vision {

class FASTDEPLOY_DECL ProcessorManager {
 public:
  ~ProcessorManager() {
    std::cout << "~processor manager" << std::endl;
#ifdef WITH_GPU
    if (stream_) cudaStreamDestroy(stream_);
#endif
  }

  void UseCuda(int gpu_id = -1) {
#ifdef WITH_GPU
    if (gpu_id >= 0) {
      device_id_ = gpu_id;
      FDASSERT(cudaSetDevice(device_id_) == cudaSuccess,
               "[ERROR] Error occurs while setting cuda device.");
    }
    FDASSERT(cudaStreamCreate(&stream_) == cudaSuccess,
             "[ERROR] Error occurs while creating cuda stream.");
    DefaultProcLib::default_lib = ProcLib::CUDA;
#else
  FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif
  }

  void UseCvCuda(int gpu_id = -1) {
#ifdef ENABLE_CVCUDA
    UseCuda(gpu_id);
    DefaultProcLib::default_lib = ProcLib::CVCUDA;
#else
    FDASSERT(false, "FastDeploy didn't compile with CV-CUDA.");
#endif
  }

  bool WithGpu() {
    return (DefaultProcLib::default_lib == ProcLib::CUDA ||
            DefaultProcLib::default_lib == ProcLib::CVCUDA);
  }

  void SetStream(Mat* mat) {
    mat->SetStream(stream_);
  }

  void SyncStream() {
#ifdef WITH_GPU
    FDASSERT(cudaStreamSynchronize(stream_) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
#endif
  }

  int DeviceId() { return device_id_; }

 private:
#ifdef WITH_GPU
  cudaStream_t stream_ = nullptr;
  int device_id_ = -1;
#endif
};

}  // namespace vision
}  // namespace fastdeploy
