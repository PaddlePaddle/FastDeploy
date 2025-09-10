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
#include "paddle/extension.h"

namespace paddle {
template <typename T>
T get_env(const std::string& var_name, T default_value);
}

#define XPU_DECLARE_VALUE(type, env_name, default_value) \
  static type FLAGS_##env_name =                         \
      paddle::get_env<type>(#env_name, default_value);

#define XPU_DECLARE_BOOL(env_name, default_value) \
  XPU_DECLARE_VALUE(bool, env_name, default_value)
#define XPU_DECLARE_INT(env_name, default_value) \
  XPU_DECLARE_VALUE(int, env_name, default_value)
