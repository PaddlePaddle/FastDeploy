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
#include "fastdeploy/runtime.h"

namespace fastdeploy {

class FASTDEPLOY_DECL FastDeployModel {
 public:
  virtual std::string ModelName() const { return "NameUndefined"; }

  virtual bool InitRuntime();
  virtual bool CreateCpuBackend();
  virtual bool CreateGpuBackend();
  virtual bool Infer(std::vector<FDTensor>& input_tensors,
                     std::vector<FDTensor>* output_tensors);

  RuntimeOption runtime_option;
  std::vector<Backend> valid_cpu_backends = {Backend::ORT};
  std::vector<Backend> valid_gpu_backends = {Backend::ORT};
  std::vector<Backend> valid_external_backends;
  bool initialized = false;
  virtual int NumInputsOfRuntime() { return runtime_->NumInputs(); }
  virtual int NumOutputsOfRuntime() { return runtime_->NumOutputs(); }
  virtual TensorInfo InputInfoOfRuntime(int index) {
    return runtime_->GetInputInfo(index);
  }
  virtual TensorInfo OutputInfoOfRuntime(int index) {
    return runtime_->GetOutputInfo(index);
  }
  virtual bool Initialized() const {
    return runtime_initialized_ && initialized;
  }

  virtual void EnableRecordTimeOfRuntime() {
    time_of_runtime_.clear();
    std::vector<double>().swap(time_of_runtime_);
    enable_record_time_of_runtime_ = true;
  }

  virtual void DisableRecordTimeOfRuntime() {
    enable_record_time_of_runtime_ = false;
  }

  virtual std::map<std::string, float> PrintStatisInfoOfRuntime();

 private:
  std::unique_ptr<Runtime> runtime_;
  bool runtime_initialized_ = false;
  // whether to record inference time
  bool enable_record_time_of_runtime_ = false;

  // record inference time for backend
  std::vector<double> time_of_runtime_;
};

}  // namespace fastdeploy
