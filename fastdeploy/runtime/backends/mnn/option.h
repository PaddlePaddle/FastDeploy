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
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace fastdeploy {

enum MNNPowerMode {
  MNN_POWER_NORMAL = 0,
  MNN_POWER_HIGH,
  MNN_POWER_LOW
};

enum MNNPrecisionMode {
  MNN_PRECISION_Normal = 0,  ///< Automatically select int8/fp32
  MNN_PRECISION_HIGH,        ///< Inference in fp32(cpu)
  MNN_PRECISION_LOW,         ///< Inference in fp16(cpu)
  MNN_PRECISION_LOW_BF16     ///< Inference in bf16(cpu)
};

struct MNNBackendOption {};

}  // namespace fastdeploy
