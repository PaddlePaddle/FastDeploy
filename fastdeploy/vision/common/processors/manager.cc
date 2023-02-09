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
  DefaultProcLib::default_lib = ProcLib::CUDA;
#else
  FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif

  if (enable_cv_cuda) {
#ifdef ENABLE_CVCUDA
    DefaultProcLib::default_lib = ProcLib::CVCUDA;
#else
    FDASSERT(false, "FastDeploy didn't compile with CV-CUDA.");
#endif
  }

  img_decoder_ = ImageDecoder(ImageDecoderLib::NVJPEG);
}

bool ProcessorManager::CudaUsed() {
  return (DefaultProcLib::default_lib == ProcLib::CUDA ||
          DefaultProcLib::default_lib == ProcLib::CVCUDA);
}

bool ProcessorManager::Run(std::vector<FDMat>* images,
                           std::vector<FDTensor>* outputs) {
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }

  if (images->size() > input_caches_.size()) {
    input_caches_.resize(images->size());
    output_caches_.resize(images->size());
  }

  FDMatBatch image_batch(images);
  image_batch.input_cache = &batch_input_cache_;
  image_batch.output_cache = &batch_output_cache_;

  for (size_t i = 0; i < images->size(); ++i) {
    if (CudaUsed()) {
      SetStream(&image_batch);
    }
    (*images)[i].input_cache = &input_caches_[i];
    (*images)[i].output_cache = &output_caches_[i];
  }

  bool ret = Apply(&image_batch, outputs);

  if (CudaUsed()) {
    SyncStream();
  }
  return ret;
}

bool ProcessorManager::Run(const std::vector<std::string>& img_names,
                           std::vector<FDTensor>* outputs) {
  std::vector<FDMat> mats(img_names.size());

  if (mats.size() > output_caches_.size()) {
    output_caches_.resize(mats.size());
  }

  for (size_t i = 0; i < mats.size(); ++i) {
    if (CudaUsed()) {
      SetStream(&mats[i]);
    }
    mats[i].output_cache = &output_caches_[i];
  }
  img_decoder_.BatchDecode(img_names, &mats);
  return Run(&mats, outputs);
}

}  // namespace vision
}  // namespace fastdeploy
