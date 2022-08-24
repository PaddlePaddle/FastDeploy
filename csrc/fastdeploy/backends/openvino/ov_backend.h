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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fastdeploy/backends/backend.h"
#include "openvino/openvino.hpp"

namespace fastdeploy {

struct OpenVINOBackendOption {
  int cpu_thread_num = 8;
  std::map<std::string, std::vector<int64_t>> shape_infos;
};

class OpenVINOBackend : public BaseBackend {
 public:
  OpenVINOBackend() {}
  virtual ~OpenVINOBackend() = default;

  bool
  InitFromPaddle(const std::string& model_file, const std::string& params_file,
                 const OpenVINOBackendOption& option = OpenVINOBackendOption());

  bool
  InitFromOnnx(const std::string& model_file,
               const OpenVINOBackendOption& option = OpenVINOBackendOption());

  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs() const;

  int NumOutputs() const;

  TensorInfo GetInputInfo(int index);
  TensorInfo GetOutputInfo(int index);

 private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest request_;
  OpenVINOBackendOption option_;
  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
};
}  // namespace fastdeploy
