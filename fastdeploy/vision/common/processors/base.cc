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
#include "fastdeploy/vision/common/processors/proc_lib.h"

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
  } else if (target == ProcLib::CVCUDA) {
#ifdef ENABLE_CVCUDA
    if (mat->Stream() == nullptr) {
      FDWARNING
          << "When using CV-CUDA, it's better to create mat->stream externally."
          << std::endl;
      cudaStream_t stream;
      FDASSERT(cudaStreamCreate(&stream) == 0,
               "[ERROR] Error occurs while calling cudaStreamCreate().");
      mat->SetStream(stream);
    }
    return ImplByCvCuda(mat);
#else
    FDASSERT(false, "FastDeploy didn't compile with CV-CUDA.");
#endif
  }
  // DEFAULT & OPENCV
  mat->MakeSureOnCpu();
  return ImplByOpenCV(mat);
}

FDTensor* Processor::UpdateAndGetReusedTensor(
    const std::vector<int64_t>& new_shape, const FDDataType& data_type,
    const std::string& tensor_name, const Device& new_device,
    const bool& use_pinned_memory) {
  if (reused_tensors_.count(tensor_name) == 0) {
    reused_tensors_[tensor_name] = FDTensor();
  }
  reused_tensors_[tensor_name].is_pinned_memory = use_pinned_memory;
  reused_tensors_[tensor_name].Resize(new_shape, data_type, tensor_name,
                                      new_device);
  return &reused_tensors_[tensor_name];
}

FDTensor* Processor::CreateCachedGpuInputTensor(
    Mat* mat, const std::string& tensor_name) {
#ifdef WITH_GPU
  FDTensor* src = mat->Tensor();
  if (src->device == Device::GPU) {
    return src;
  } else if (src->device == Device::CPU) {
    FDTensor* tensor = UpdateAndGetReusedTensor(src->Shape(), src->Dtype(),
                                                tensor_name, Device::GPU);
    FDASSERT(cudaMemcpyAsync(tensor->Data(), src->Data(), tensor->Nbytes(),
                             cudaMemcpyHostToDevice, mat->Stream()) == 0,
             "[ERROR] Error occurs while copy memory from CPU to GPU.");
    return tensor;
  } else {
    FDASSERT(false, "FDMat is on unsupported device: %d", src->device);
  }
#else
  FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif
  return nullptr;
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
