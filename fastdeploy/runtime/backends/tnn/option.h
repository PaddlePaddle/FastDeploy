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

#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/runtime/enum_variables.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace fastdeploy {

struct TNNBackendOption {
  /// Number of threads while use CPU
  int cpu_threads = 1;
  /// CPU power mode: include/tnn/utils/cpu_utils.h
  /// bind all cores: 0, bind little cores: 1, bind big cores: 2
  int cpu_powersave = 0;
  /// Inference precision: include/tnn/core/common.h
  // --- Auto precision, PRECISION_AUTO = -1: FP16
  // each device choose default precision.
  // ARM: prefer fp16, enable approximate calculation
  // --- Normal precision, PRECISION_NORMAL = 0: FP16
  // ARM: prefer fp16, disable approximate calculation
  // --- High precision, PRECISION_HIGH = 1: FP32
  // ARM: run with fp32
  // --- Low precision, PRECISION_LOW = 2: BF16
  // ARM: run with bfp16
  /// Enable use int8 precision
  bool enable_int8 = false;
  /// Enable use fp16 precision
  bool enable_fp16 = false;
  /// Enable use bf16 precision
  bool enable_bf16 = false;
  /// Inference device, TNN support CPU/GPU(OpenCL/VULKAN)
  Device device = Device::CPU;
  /// Index of inference device
  int device_id = 0;
  /// Custom orders of input tensors
  std::map<std::string, int> in_orders{};
  /// Custom orders of output tensors
  std::map<std::string, int> out_orders{};
};

}  // namespace fastdeploy
