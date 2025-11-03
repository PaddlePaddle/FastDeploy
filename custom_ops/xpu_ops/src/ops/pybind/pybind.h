// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda_runtime_api.h>
#include <xpu/runtime.h>
#include <exception>
#include "ops/pybind/cachekv_signal_thread_worker.h"

// 自定义异常类，用于处理XPU错误
class XPUError : public std::exception {
 public:
  explicit XPUError(int error) : error_(error) {}

  const char *what() const noexcept override { return xpu_strerror(error_); }

 private:
  int error_;
};
