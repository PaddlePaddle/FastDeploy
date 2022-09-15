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
    if (option.enable_trt) {
#ifdef ENABLE_TRT_BACKEND
      auto precision = paddle_infer::PrecisionType::kFloat32;
      if (option.trt_option.enable_fp16) {
        precision = paddle_infer::PrecisionType::kHalf;
      }
      bool use_static = false;
      if (option.trt_option.serialize_file != "") {
        FDWARNING << "Detect that tensorrt cache file has been set to " << option.trt_option.serialize_file << ", but while enable paddle2trt, please notice that the cache file will save to the directory where paddle model saved." << std::endl;
        use_static = true;
      }
      config_.EnableTensorRtEngine(option.trt_option.max_workspace_size, 32, 3, precision, use_static);
      std::map<std::string, std::vector<int>> max_shape;
      std::map<std::string, std::vector<int>> min_shape;
      std::map<std::string, std::vector<int>> opt_shape;
      for (const auto& item : option.trt_option.min_shape) {
        auto max_iter = option.trt_option.max_shape.find(item.first);
        auto opt_iter = option.trt_option.opt_shape.find(item.first);
        FDASSERT(max_iter != option.trt_option.max_shape.end(), "Cannot find %s in TrtBackendOption::min_shape.", item.first.c_str());
        FDASSERT(opt_iter != option.trt_option.opt_shape.end(), "Cannot find %s in TrtBackendOption::opt_shape.", item.first.c_str());
        max_shape[item.first].assign(max_iter->second.begin(), max_iter->second.end());
        opt_shape[item.first].assign(opt_iter->second.begin(), opt_iter->second.end());
        min_shape[item.first].assign(item.second.begin(), item.second.end());
      }
      if (min_shape.size() > 0) {
        config_.SetTRTDynamicShapeInfo(min_shape, max_shape, opt_shape);
      }
#else
      FDWARING << "The FastDeploy is not compiled with TensorRT backend, so will fallback to GPU with Paddle Inference Backend." << std::endl;
#endif
    }
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
  if (!option.delete_pass_names.empty()) {
    auto pass_builder = config_.pass_builder();
    for (int i = 0; i < option.delete_pass_names.size(); i++) {
      FDINFO << "Delete pass : " << option.delete_pass_names[i] << std::endl;
      pass_builder->DeletePass(option.delete_pass_names[i]);
    }
  }
  if (option.cpu_thread_num <= 0) {
    config_.SetCpuMathLibraryNumThreads(8);
  } else {
    config_.SetCpuMathLibraryNumThreads(option.cpu_thread_num);
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

std::vector<TensorInfo> PaddleBackend::GetInputInfo() { return inputs_desc_; }

TensorInfo PaddleBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetOutputInfo() { return outputs_desc_; }

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
    ShareTensorFromFDTensor(handle.get(), inputs[i]);
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
