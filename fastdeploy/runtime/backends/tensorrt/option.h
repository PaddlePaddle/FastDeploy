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
#include "fastdeploy/core/fd_type.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fastdeploy {

struct TrtBackendOption {
  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty
  // format of input model
  ModelFormat model_format = ModelFormat::AUTOREC;

  int gpu_id = 0;
  bool enable_fp16 = false;
  bool enable_int8 = false;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
  std::map<std::string, std::vector<int32_t>> max_shape;
  std::map<std::string, std::vector<int32_t>> min_shape;
  std::map<std::string, std::vector<int32_t>> opt_shape;
  std::string serialize_file = "";
  bool enable_pinned_memory = false;
  void* external_stream_ = nullptr;
};
}  // namespace fastdeploy
