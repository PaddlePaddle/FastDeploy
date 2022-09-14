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

#include "fastdeploy/backends/common/multiclass_nms.h"
#include "fastdeploy/core/fd_tensor.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fastdeploy {

struct TensorInfo {
  std::string name;
  std::vector<int> shape;
  FDDataType dtype;

  friend std::ostream& operator<<(std::ostream& output,
                                  const TensorInfo& info) {
    output << "TensorInfo(name: " << info.name << ", shape: [";
    for (size_t i = 0; i < info.shape.size(); ++i) {
      if (i == info.shape.size() - 1) {
        output << info.shape[i];
      } else {
        output << info.shape[i] << ", ";
      }
    }
    output << "], dtype: " << Str(info.dtype) << ")";
    return output;
  }
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
