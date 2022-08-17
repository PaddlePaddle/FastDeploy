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

#include "fastdeploy/backends/paddle/paddle_backend.h"

namespace fastdeploy {

void PaddleBackend::BuildOption(const PaddleBackendOption& option) {
  if (option.use_gpu) {
    config_.EnableUseGpu(option.gpu_mem_init_size, option.gpu_id);
  } else {
    config_.DisableGpu();
    if (option.enable_mkldnn) {
      config_.EnableMKLDNN();
      config_.SetMkldnnCacheCapacity(option.mkldnn_cache_size);
    }
  }
  if (!option.enable_log_info) {
    config_.DisableGlogInfo();
  }
  config_.SetCpuMathLibraryNumThreads(option.cpu_thread_num);

  if (!option.delete_pass_names.empty()) {
    auto pass_builder = config_.pass_builder();
    for (int i = 0; i < option.delete_pass_names.size(); i++) {
      std::cout << "Delete pass : " << option.delete_pass_names[i] << std::endl;
      pass_builder->DeletePass(option.delete_pass_names[i]);
    }
  }
}

bool PaddleBackend::InitFromPaddle(const std::string& model_file,
                                   const std::string& params_file,
                                   const PaddleBackendOption& option) {
  if (initialized_) {
    FDERROR << "PaddleBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  config_.SetModel(model_file, params_file);
  BuildOption(option);
  predictor_ = paddle_infer::CreatePredictor(config_);
  std::vector<std::string> input_names = predictor_->GetInputNames();
  std::vector<std::string> output_names = predictor_->GetOutputNames();
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto handle = predictor_->GetInputHandle(input_names[i]);
    TensorInfo info;
    auto shape = handle->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.dtype = PaddleDataTypeToFD(handle->type());
    info.name = input_names[i];
    inputs_desc_.emplace_back(info);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(output_names[i]);
    TensorInfo info;
    auto shape = handle->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.dtype = PaddleDataTypeToFD(handle->type());
    info.name = output_names[i];
    outputs_desc_.emplace_back(info);
  }
  initialized_ = true;
  return true;
}

TensorInfo PaddleBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

TensorInfo PaddleBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

bool PaddleBackend::Infer(std::vector<FDTensor>& inputs,
                          std::vector<FDTensor>* outputs) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[PaddleBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto handle = predictor_->GetInputHandle(inputs[i].name);
    ShareTensorFromCpu(handle.get(), inputs[i]);
  }

  predictor_->Run();
  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(outputs_desc_[i].name);
    CopyTensorToCpu(handle, &((*outputs)[i]));
  }
  return true;
}

}  // namespace fastdeploy
