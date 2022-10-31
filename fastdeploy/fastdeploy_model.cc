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
      CheckModelFormat(runtime_option.model_file, runtime_option.model_format),
      "ModelFormatCheck Failed.");
  if (runtime_initialized_) {
    FDERROR << "The model is already initialized, cannot be initliazed again."
            << std::endl;
    return false;
  }
  if (runtime_option.backend != Backend::UNKNOWN) {
    if (!IsBackendAvailable(runtime_option.backend)) {
      FDERROR << Str(runtime_option.backend)
              << " is not compiled with current FastDeploy library."
              << std::endl;
      return false;
    }

    bool use_gpu = (runtime_option.device == Device::GPU);
#ifndef WITH_GPU
    use_gpu = false;
#endif

    // whether the model is supported by the setted backend
    bool is_supported = false;
    if (use_gpu) {
      for (auto& item : valid_gpu_backends) {
        if (item == runtime_option.backend) {
          is_supported = true;
          break;
        }
      }
    } else {
      for (auto& item : valid_cpu_backends) {
        if (item == runtime_option.backend) {
          is_supported = true;
          break;
        }
      }
    }

    if (is_supported) {
      runtime_ = std::shared_ptr<Runtime>(new Runtime());
      if (!runtime_->Init(runtime_option)) {
        return false;
      }
      runtime_initialized_ = true;
      return true;
    } else {
      FDWARNING << ModelName() << " is not supported with backend "
                << Str(runtime_option.backend) << "." << std::endl;
      if (use_gpu) {
        FDASSERT(valid_gpu_backends.size() > 0,
                 "There's no valid gpu backend for %s.", ModelName().c_str());
        FDWARNING << "FastDeploy will choose " << Str(valid_gpu_backends[0])
                  << " for model inference." << std::endl;
      } else {
        FDASSERT(valid_cpu_backends.size() > 0,
                 "There's no valid cpu backend for %s.", ModelName().c_str());
        FDWARNING << "FastDeploy will choose " << Str(valid_cpu_backends[0])
                  << " for model inference." << std::endl;
      }
    }
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

  for (size_t i = 0; i < valid_cpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_cpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_cpu_backends[i];
    runtime_ = std::shared_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Found no valid backend for model: " << ModelName() << std::endl;
  return false;
}

bool FastDeployModel::CreateGpuBackend() {
  if (valid_gpu_backends.size() == 0) {
    FDERROR << "There's no valid gpu backends for model: " << ModelName()
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < valid_gpu_backends.size(); ++i) {
    if (!IsBackendAvailable(valid_gpu_backends[i])) {
      continue;
    }
    runtime_option.backend = valid_gpu_backends[i];
    runtime_ = std::shared_ptr<Runtime>(new Runtime());
    if (!runtime_->Init(runtime_option)) {
      return false;
    }
    runtime_initialized_ = true;
    return true;
  }
  FDERROR << "Cannot find an available gpu backend to load this model."
          << std::endl;
  return false;
}

bool FastDeployModel::Infer(std::vector<FDTensor>& _input_tensors,
                            std::vector<FDTensor>* _output_tensors) {
  TimeCounter tc;
  if (enable_record_time_of_runtime_) {
    tc.Start();
  }
  auto ret = runtime_->Infer(_input_tensors, _output_tensors);
  if (enable_record_time_of_runtime_) {
    tc.End();
    if (time_of_runtime_.size() > 50000) {
      FDWARNING << "There are already 50000 records of runtime, will force to "
                   "disable record time of runtime now."
                << std::endl;
      enable_record_time_of_runtime_ = false;
    }
    time_of_runtime_.push_back(tc.Duration());
  }
  return ret;
}

bool FastDeployModel::Infer() {
  return Infer(input_tensors, &output_tensors);
}

std::map<std::string, float> FastDeployModel::PrintStatisInfoOfRuntime() {
  std::map<std::string, float> statis_info_of_runtime_dict;

  if (time_of_runtime_.size() < 10) {
    FDWARNING << "PrintStatisInfoOfRuntime require the runtime ran 10 times at "
                 "least, but now you only ran "
              << time_of_runtime_.size() << " times." << std::endl;
  }
  double warmup_time = 0.0;
  double remain_time = 0.0;
  int warmup_iter = time_of_runtime_.size() / 5;
  for (size_t i = 0; i < time_of_runtime_.size(); ++i) {
    if (i < warmup_iter) {
      warmup_time += time_of_runtime_[i];
    } else {
      remain_time += time_of_runtime_[i];
    }
  }
  double avg_time = remain_time / (time_of_runtime_.size() - warmup_iter);
  std::cout << "============= Runtime Statis Info(" << ModelName()
            << ") =============" << std::endl;
  std::cout << "Total iterations: " << time_of_runtime_.size() << std::endl;
  std::cout << "Total time of runtime: " << warmup_time + remain_time << "s."
            << std::endl;
  std::cout << "Warmup iterations: " << warmup_iter << std::endl;
  std::cout << "Total time of runtime in warmup step: " << warmup_time << "s."
            << std::endl;
  std::cout << "Average time of runtime exclude warmup step: "
            << avg_time * 1000 << "ms." << std::endl;

  statis_info_of_runtime_dict["total_time"] = warmup_time + remain_time;
  statis_info_of_runtime_dict["warmup_time"] = warmup_time;
  statis_info_of_runtime_dict["remain_time"] = remain_time;
  statis_info_of_runtime_dict["warmup_iter"] = warmup_iter;
  statis_info_of_runtime_dict["avg_time"] = avg_time;
  statis_info_of_runtime_dict["iterations"] = time_of_runtime_.size();
  return statis_info_of_runtime_dict;
}
}  // namespace fastdeploy
