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
#include "fastdeploy/vision/common/processors/manager.h"

namespace fastdeploy {
namespace vision {

ProcessorManager::~ProcessorManager() {
#ifdef WITH_GPU
  if (stream_) cudaStreamDestroy(stream_);
#endif
}

void ProcessorManager::UseCuda(bool enable_cv_cuda, int gpu_id) {
#ifdef WITH_GPU
  if (gpu_id >= 0) {
    device_id_ = gpu_id;
    FDASSERT(cudaSetDevice(device_id_) == cudaSuccess,
             "[ERROR] Error occurs while setting cuda device.");
  }
  FDASSERT(cudaStreamCreate(&stream_) == cudaSuccess,
           "[ERROR] Error occurs while creating cuda stream.");
  proc_lib_ = ProcLib::CUDA;
#else
  FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif

  if (enable_cv_cuda) {
#ifdef ENABLE_CVCUDA
    proc_lib_ = ProcLib::CVCUDA;
#else
    FDASSERT(false, "FastDeploy didn't compile with CV-CUDA.");
#endif
  }
}

bool ProcessorManager::CudaUsed() {
  return (proc_lib_ == ProcLib::CUDA || proc_lib_ == ProcLib::CVCUDA);
}

bool ProcessorManager::PreApply(FDMatBatch* image_batch) {
  if (image_batch->mats->size() == 0) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  
  if (image_batch->mats->size() > input_caches_.size()) {
    input_caches_.resize(image_batch->mats->size());
    output_caches_.resize(image_batch->mats->size());
  }

  image_batch->input_cache = &batch_input_cache_;
  image_batch->output_cache = &batch_output_cache_;

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    if (CudaUsed()) {
      SetStream(image_batch);
    }
    (*(image_batch->mats))[i].input_cache = &input_caches_[i];
    (*(image_batch->mats))[i].output_cache = &output_caches_[i];
    if ((*(image_batch->mats))[i].mat_type == ProcLib::CUDA) {
      // Make a copy of the input data ptr, so that the original data ptr of
      // FDMat won't be modified.
      auto fd_tensor = std::make_shared<FDTensor>();
      fd_tensor->SetExternalData(
          (*(image_batch->mats))[i].Tensor()->shape, (*(image_batch->mats))[i].Tensor()->Dtype(),
          (*(image_batch->mats))[i].Tensor()->Data(), (*(image_batch->mats))[i].Tensor()->device,
          (*(image_batch->mats))[i].Tensor()->device_id);
      (*(image_batch->mats))[i].SetTensor(fd_tensor);
    }
  }
  return true;
}

bool ProcessorManager::PostApply() {
  if (CudaUsed()) {
    SyncStream();
  }
  return true;
}

bool ProcessorManager::Run(std::vector<FDMat>* images,
                           std::vector<FDTensor>* outputs) {

  FDMatBatch image_batch(images);
  bool preApply = PreApply(&image_batch);

  bool ret = Apply(&image_batch, outputs);

  bool postApply = PostApply();
  
  return ret;
}

}  // namespace vision
}  // namespace fastdeploy
