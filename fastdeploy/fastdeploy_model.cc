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
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

bool FastDeployModel::InitRuntime() {
  FDASSERT(
      ModelFormatCheck(runtime_option.model_file, runtime_option.model_format),
      "ModelFormatCheck Failed.");
  if (runtime_initialized_) {
    FDERROR << "The model is already initialized, cannot be initliazed again."
            << std::endl;
    return false;
  }
  if (runtime_option.backend != Backend::UNKNOWN) {
    if (runtime_option.backend == Backend::ORT) {
      if (!IsBackendAvailable(Backend::ORT)) {
        FDERROR
            << "Backend::ORT is not complied with current FastDeploy library."
            << std::endl;
        return false;
      }
    } else if (runtime_option.backend == Backend::TRT) {
      if (!IsBackendAvailable(Backend::TRT)) {
        FDERROR
            << "Backend:TRT is not complied with current FastDeploy library."
            << std::endl;
        return false;
      }
    } else {
      FDERROR << "Only support Backend::ORT / Backend::TRT now." << std::endl;
      return false;
    }
    runtime_ = new Runtime();
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }

  if (runtime_option.device == Device::CPU) {
    return CreateCpuBackend();
  } else if (runtime_option.device == Device::GPU) {
#ifdef WITH_GPU
    return CreateGpuBackend();
#else
    FDERROR << "The compiled FastDeploy library doesn't support GPU now."
            << std::endl;
    return false;
#endif
  }
  FDERROR << "Only support CPU/GPU now." << std::endl;
  return false;
}

bool FastDeployModel::CreateCpuBackend() {
  if (valid_cpu_backends.size() == 0) {
    FDERROR << "There's no valid cpu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (auto& b : valid_cpu_backends) {
    if (b == Backend::ORT) {
      if (!IsBackendAvailable(Backend::ORT)) {
        FDERROR << "OrtBackend is not complied with current FastDeploy library."
                << std::endl;
        continue;
      }
      runtime_option.backend = b;
      runtime_ = new Runtime();
      if (!runtime_->Init(runtime_option)) {
        return false;
      }
      runtime_initialized_ = true;
      return true;
    } else {
      FDERROR << "Only Backend::ORT as cpu backend is supported now."
              << std::endl;
      return false;
    }
  }

  FDERROR << "Cannot find an available cpu backend to load this model."
          << std::endl;
  return false;
}

bool FastDeployModel::CreateGpuBackend() {
  if (valid_gpu_backends.size() == 0) {
    FDERROR << "There's no valid gpu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (auto& b : valid_gpu_backends) {
    if (b == Backend::ORT) {
      if (!IsBackendAvailable(Backend::ORT)) {
        FDERROR << "OrtBackend is not complied with current FastDeploy library."
                << std::endl;
        continue;
      }
      runtime_option.backend = b;
      runtime_ = new Runtime();
      if (!runtime_->Init(runtime_option)) {
        return false;
      }
      runtime_initialized_ = true;
      return true;
    } else if (b == Backend::TRT) {
      if (!IsBackendAvailable(Backend::TRT)) {
        FDERROR << "TrtBackend is not complied with current FastDeploy library."
                << std::endl;
        continue;
      }
      runtime_option.backend = b;
      runtime_ = new Runtime();
      if (!runtime_->Init(runtime_option)) {
        return false;
      }
      return true;
    } else {
      FDERROR << "Only Backend::ORT / Backend::TRT as gpu backends are "
                 "supported now."
              << std::endl;
      return false;
    }
  }

  FDERROR << "Cannot find an available gpu backend to load this model."
          << std::endl;
  return false;
}

bool FastDeployModel::Infer(std::vector<FDTensor>& input_tensors,
                            std::vector<FDTensor>* output_tensors) {
  return runtime_->Infer(input_tensors, output_tensors);
}

void FastDeployModel::EnableDebug() {
#ifdef FASTDEPLOY_DEBUG
  debug_ = true;
#else
  FDLogger() << "The compile FastDeploy is not with -DENABLE_DEBUG=ON, so "
                "cannot enable debug mode."
             << std::endl;
  debug_ = false;
#endif
}

bool FastDeployModel::DebugEnabled() { return debug_; }

} // namespace fastdeploy
