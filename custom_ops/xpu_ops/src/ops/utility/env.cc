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

#include "env.h"  // NOLINT

namespace paddle {

// Specialization for bool
template <>
bool get_env<bool>(const std::string& var_name, bool default_value) {
  const char* value = std::getenv(var_name.c_str());
  if (!value) {
    if (var_name.size() < 6 || var_name.substr(0, 6) != "FLAGS_") {
      return get_env<bool>("FLAGS_" + var_name, default_value);
    }
    return default_value;
  }
  std::string valStr(value);
  std::transform(valStr.begin(), valStr.end(), valStr.begin(), ::tolower);
  if (valStr == "true" || valStr == "1") {
    return true;
  } else if (valStr == "false" || valStr == "0") {
    return false;
  }
  PD_THROW("Unexpected value:", valStr, ", only bool supported.");
  return default_value;
}

template <>
int get_env<int>(const std::string& var_name, int default_value) {
  const char* value = std::getenv(var_name.c_str());
  if (!value) {
    if (var_name.size() < 6 || var_name.substr(0, 6) != "FLAGS_") {
      return get_env<int>("FLAGS_" + var_name, default_value);
    }
    return default_value;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    PD_THROW("Unexpected value:", value, ", only int supported.");
  }
}

#define DEFINE_GET_ENV_SPECIALIZATION(T) \
  template <>                            \
  T get_env<T>(const std::string& var_name, T default_value);

DEFINE_GET_ENV_SPECIALIZATION(bool)
DEFINE_GET_ENV_SPECIALIZATION(int)

}  // namespace paddle
