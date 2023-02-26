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

/*! Power mode for MNN BackendConfig. */
enum MNNPowerMode {
  MNN_POWER_NORMAL = 0,
  MNN_POWER_HIGH,
  MNN_POWER_LOW
};
/*! Precisio mode for MNN BackendConfig. */
enum MNNPrecisionMode {
  MNN_PRECISION_Normal = 0,  ///< Automatically select int8/fp32
  MNN_PRECISION_HIGH,        ///< Inference in fp32(cpu)
  MNN_PRECISION_LOW,         ///< Inference in fp16(cpu)
  MNN_PRECISION_LOW_BF16     ///< Inference in bf16(cpu)
};

struct MNNBackendOption {
  /// MNN power mode for mobile device.
  MNNPowerMode power_mode = MNN_POWER_NORMAL;
  /// Number of threads while use CPU
  int cpu_threads = 1;
  /// Enable use half precision
  bool enable_fp16 = false;
  /// Inference device, MNN support CPU/GPU
  Device device = Device::CPU;
  /// Index of inference device
  int device_id = 0;
  /// Custom orders of input tensors
  std::map<std::string, int> in_orders{};
  /// Custom orders of output tensors
  std::map<std::string, int> out_orders{};
  /// Custom fixed shape for input tensors
  std::map<std::string, std::vector<int>> in_shapes{};
  /// Custom tensors to keep to avoid memory reuse
  /// Useful for benchmark profile to keep the values of
  /// input tensors.
  std::vector<std::string> save_tensors{};
};
}  // namespace fastdeploy
