// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/runtime/backends/mnn/option.h"

namespace fastdeploy {

class MNNBackend : public BaseBackend {
 public:
  MNNBackend() {}
  ~MNNBackend() override;

  bool Init(const RuntimeOption& runtime_option) override;

  bool Infer(std::vector<FDTensor>& inputs,
             std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override; // NOLINT

  int NumInputs() const override { return inputs_desc_.size(); }

  int NumOutputs() const override { return outputs_desc_.size(); }

  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;

 private:
  void BuildOption(const MNNBackendOption& option);
  bool UpdateInputShapeAndDesc(const std::vector<FDTensor>& inputs);

  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
  std::map<std::string, int> inputs_order_;
  std::map<std::string, int> outputs_order_;
  MNNBackendOption option_;

  // MNN Interpreter and Session
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_{nullptr};
  MNN::ScheduleConfig schedule_config_;
  MNN::BackendConfig backend_config_;
};

// Convert data type from MNN to fastdeploy
FDDataType MNNDataTypeToFD(const halide_type_t& dtype);

}  // namespace fastdeploy
