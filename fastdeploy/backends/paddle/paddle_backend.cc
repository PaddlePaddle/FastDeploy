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

  // The input/output information get from predictor is not right, use PaddleReader instead now
  std::string contents;
  if (!ReadBinaryFromFile(model_file, &contents)) {
    return false;
  }
  auto reader =
      paddle2onnx::PaddleReader(contents.c_str(), contents.size());

  // If it's a quantized model, and use cpu with mkldnn, automaticaly switch to int8 mode
  if (reader.is_quantize_model) {
    if (option.use_gpu) {
      FDWARNING << "The loaded model is a quantized model, while inference on GPU, please use TensorRT backend to get better performance." << std::endl;
    }
    if (option.enable_mkldnn) {
      config_.EnableMkldnnInt8();
    } else {
      FDWARNING << "The loaded model is a quantized model, while inference on CPU, please enable MKLDNN to get better performance." << std::endl;
    }
  }

  inputs_desc_.resize(reader.num_inputs);
  for (int i = 0; i < reader.num_inputs; ++i) {
    std::string name(reader.inputs[i].name);
    std::vector<int64_t> shape(
        reader.inputs[i].shape,
        reader.inputs[i].shape + reader.inputs[i].rank);
    inputs_desc_[i].name = name;
    inputs_desc_[i].shape.assign(shape.begin(), shape.end());
    inputs_desc_[i].dtype = ReaderDataTypeToFD(reader.inputs[i].dtype);
  }
  outputs_desc_.resize(reader.num_outputs);
  for (int i = 0; i < reader.num_outputs; ++i) {
    std::string name(reader.outputs[i].name);
    std::vector<int64_t> shape(reader.outputs[i].shape, reader.outputs[i].shape + reader.outputs[i].rank);
    outputs_desc_[i].name = name;
    outputs_desc_[i].shape.assign(shape.begin(), shape.end());
    outputs_desc_[i].dtype = ReaderDataTypeToFD(reader.outputs[i].dtype);
  }

  predictor_ = paddle_infer::CreatePredictor(config_);
  initialized_ = true;
  return true;
}

TensorInfo PaddleBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo PaddleBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetOutputInfos() {
  return outputs_desc_;
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
