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

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/runtime/backends/backend.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <unistd.h>

namespace fastdeploy {
class TVMBackend : public BaseBackend {
 public:
  TVMBackend() = default;
  virtual ~TVMBackend() = default;
  bool Init(const RuntimeOption& runtime_option) override;
  int NumInputs() const override { return -1; }
  int NumOutputs() const override { return -1; }
  TensorInfo GetInputInfo(int index) override { return TensorInfo{}; }
  TensorInfo GetOutputInfo(int index) override { return TensorInfo{}; }
  std::vector<TensorInfo> GetInputInfos() override { return {TensorInfo{}}; }
  std::vector<TensorInfo> GetOutputInfos() override { return {TensorInfo{}}; }
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs,
             bool copy_to_fd = true) override;

 private:
  DLDevice dev_;
  bool BuildDLDevice(Device device);
  bool BuildModel(const RuntimeOption& runtime_option);
};
}  // namespace fastdeploy
