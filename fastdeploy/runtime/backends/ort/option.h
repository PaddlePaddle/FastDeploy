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
#include <memory>
#include <string>
#include <vector>
#include <map>
namespace fastdeploy {

struct OrtBackendOption {
  // -1 means default
  // 0: ORT_DISABLE_ALL
  // 1: ORT_ENABLE_BASIC
  // 2: ORT_ENABLE_EXTENDED
  // 99: ORT_ENABLE_ALL (enable some custom optimizations e.g bert)
  int graph_optimization_level = -1;
  int intra_op_num_threads = -1;
  int inter_op_num_threads = -1;
  // 0: ORT_SEQUENTIAL
  // 1: ORT_PARALLEL
  int execution_mode = -1;
  bool use_gpu = false;
  int gpu_id = 0;
  void* external_stream_ = nullptr;

  // inside parameter, maybe remove next version
  bool remove_multiclass_nms_ = false;
  std::map<std::string, std::string> custom_op_info_;
};
}  // namespace fastdeploy
