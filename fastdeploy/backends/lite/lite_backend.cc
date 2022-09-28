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

#include "fastdeploy/backends/lite/lite_backend.h"

#include <cstring>

namespace fastdeploy {

// Convert data type from paddle lite to fastdeploy
FDDataType LiteDataTypeToFD(const paddle::lite_api::PrecisionType& dtype) {
  if (dtype == paddle::lite_api::PrecisionType::kFloat) {
    return FDDataType::FP32;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt8) {
    return FDDataType::INT8;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt32) {
    return FDDataType::INT32;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt64) {
    return FDDataType::INT64;
  } else if (dtype == paddle::lite_api::PrecisionType::kInt16) {
    return FDDataType::INT16;
  } else if (dtype == paddle::lite_api::PrecisionType::kUInt8) {
    return FDDataType::UINT8;
  } else if (dtype == paddle::lite_api::PrecisionType::kFP64) {
    return FDDataType::FP64;
  }
  FDASSERT(false, "Unexpected data type of %d.", dtype);
  return FDDataType::FP32;
}

void LiteBackend::BuildOption(const LiteBackendOption& option) {
  std::vector<paddle::lite_api::Place> valid_places;
  if (option.enable_fp16) {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kARM), PRECISION(kFP16)});
  } else {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
  }
  config_.set_valid_places(valid_places);
  if (option.threads > 0) {
    config_.set_threads(option.threads);
  }
  if (option.power_mode > 0) {
    config_.set_power_mode(
        static_cast<paddle::lite_api::PowerMode>(option.power_mode));
  }
}

bool LiteBackend::InitFromPaddle(const std::string& model_file,
                                 const std::string& params_file,
                                 const LiteBackendOption& option) {
  if (initialized_) {
    FDERROR << "LiteBackend is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }

  config_.set_model_file(model_file);
  config_.set_param_file(params_file);
  BuildOption(option);
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(
          config_);

  inputs_desc_.clear();
  outputs_desc_.clear();
  inputs_order_.clear();
  std::vector<std::string> input_names = predictor_->GetInputNames();
  std::vector<std::string> output_names = predictor_->GetOutputNames();
  for (size_t i = 0; i < input_names.size(); ++i) {
    inputs_order_[input_names[i]] = i;
    TensorInfo info;
    auto tensor = predictor_->GetInput(i);
    auto shape = tensor->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.name = input_names[i];
    info.dtype = LiteDataTypeToFD(tensor->precision());
    inputs_desc_.emplace_back(info);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    TensorInfo info;
    auto tensor = predictor_->GetOutput(i);
    auto shape = tensor->shape();
    info.shape.assign(shape.begin(), shape.end());
    info.name = output_names[i];
    info.dtype = LiteDataTypeToFD(tensor->precision());
    outputs_desc_.emplace_back(info);
  }

  initialized_ = true;
  return true;
}

TensorInfo LiteBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> LiteBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo LiteBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> LiteBackend::GetOutputInfos() { return outputs_desc_; }

bool LiteBackend::Infer(std::vector<FDTensor>& inputs,
                        std::vector<FDTensor>* outputs) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[LiteBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = inputs_order_.find(inputs[i].name);
    if (iter == inputs_order_.end()) {
      FDERROR << "Cannot find input with name:" << inputs[i].name
              << " in loaded model." << std::endl;
      return false;
    }
    auto tensor = predictor_->GetInput(iter->second);
    tensor->Resize(inputs[i].shape);
    tensor->ShareExternalMemory(const_cast<void*>(inputs[i].CpuData()),
                                inputs[i].Nbytes(),
                                paddle::lite_api::TargetType::kARM);
  }

  predictor_->Run();

  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto tensor = predictor_->GetOutput(i);
    (*outputs)[i].Resize(tensor->shape(), outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    memcpy((*outputs)[i].MutableData(), tensor->data<void>(),
           (*outputs)[i].Nbytes());
  }
  return true;
}

}  // namespace fastdeploy
