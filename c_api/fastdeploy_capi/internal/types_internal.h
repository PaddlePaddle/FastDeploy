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

#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy_capi/core/fd_type.h"
#include <memory>

#ifdef ENABLE_VISION
#include "fastdeploy_capi/vision/types_internal.h"
#endif


typedef struct FD_C_RuntimeOptionWrapper {
  std::unique_ptr<fastdeploy::RuntimeOption> runtime_option;
} FD_C_RuntimeOptionWrapper;

namespace fastdeploy {
std::unique_ptr<fastdeploy::RuntimeOption>&
FD_C_CheckAndConvertRuntimeOptionWrapper(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper);
}

#define CHECK_AND_CONVERT_FD_TYPE(TYPENAME, variable_name)                     \
  fastdeploy::FD_C_CheckAndConvert##TYPENAME(variable_name)
