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
#include "fastdeploy/backends/common/multiclass_nms.h"
#include "fastdeploy/core/fd_tensor.h"

namespace fastdeploy {

struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  FDDataType dtype;
};

class BaseBackend {
 public:
  bool initialized_ = false;

  BaseBackend() {}
  virtual ~BaseBackend() = default;

  virtual bool Initialized() const { return initialized_; }

  virtual int NumInputs() const = 0;
  virtual int NumOutputs() const = 0;
  virtual TensorInfo GetInputInfo(int index) = 0;
  virtual TensorInfo GetOutputInfo(int index) = 0;
  virtual bool Infer(std::vector<FDTensor>& inputs,
                     std::vector<FDTensor>* outputs) = 0;
};

}  // namespace fastdeploy
