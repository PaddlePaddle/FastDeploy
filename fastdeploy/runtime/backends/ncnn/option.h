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

struct NCNNBackendOption {
  /// Whether to use light mode, default false.
  bool light_mode = true;
  /// Number of threads while use CPU
  int cpu_threads = 1;
  /// CPU power mode: https://github.com/Tencent/ncnn/wiki/faq
  /// bind all cores: 0, bind little cores: 1, bind big cores: 2
  int cpu_powersave = 0;
  /// Inference options in NCNN
  /// FP16:
  ///   bool use_fp16_packed;
  ///   bool use_fp16_storage;
  ///   bool use_fp16_arithmetic;
  /// INT8:
  ///   bool use_int8_packed;
  ///   bool use_int8_storage;
  ///   bool use_int8_arithmetic;
  ///   bool use_int8_inference;
  /// BF16:
  ////  bool use_bf16_storage;
  /// Enable use int8 precision
  bool enable_int8 = false;
  /// Enable use fp16 precision
  bool enable_fp16 = false;
  /// Enable use bf16 precision
  bool enable_bf16 = false;
  /// Inference device, NCNN support CPU/GPU(OpenCV/VULKAN)
  Device device = Device::CPU;
  /// Index of inference device
  int device_id = 0;
  /// Custom orders of input tensors
  std::map<std::string, int> in_orders{};
  /// Custom orders of output tensors
  std::map<std::string, int> out_orders{};
  /// Custom output dtype hints. No explicit
  /// in NCNN Mat (only raw void pointer avaliable).
  /// Users must be know the explicit dtype of
  /// output tensors.
  std::map<std::string, FDDataType> out_dtypes{};
};

}  // namespace fastdeploy
