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
#include "fastdeploy/runtime/backends/tensorrt/option.h"


namespace fastdeploy {

struct IpuOption {
  int ipu_device_num;
  int ipu_micro_batch_size;
  bool ipu_enable_pipelining;
  int ipu_batches_per_step;
  bool ipu_enable_fp16;
  int ipu_replica_num;
  float ipu_available_memory_proportion;
  bool ipu_enable_half_partial;
};

struct PaddleBackendOption {
  std::string model_file = "";   // Path of model file
  std::string params_file = "";  // Path of parameters file, can be empty

  std::string model_buffer_ = "";
  std::string params_buffer_ = "";
  size_t model_buffer_size_ = 0;
  size_t params_buffer_size_ = 0;
  bool model_from_memory_ = false;

#ifdef WITH_GPU
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif
  bool enable_mkldnn = true;

  bool enable_log_info = false;

  bool enable_trt = false;
  TrtBackendOption trt_option;
  bool collect_shape = false;
  std::vector<std::string> trt_disabled_ops_{};

#ifdef WITH_IPU
  bool use_ipu = true;
  IpuOption ipu_option;
#else
  bool use_ipu = false;
#endif

  int mkldnn_cache_size = 1;
  int cpu_thread_num = 8;
  // initialize memory size(MB) for GPU
  int gpu_mem_init_size = 100;
  // gpu device id
  int gpu_id = 0;
  bool enable_pinned_memory = false;
  void* external_stream_ = nullptr;

  std::vector<std::string> delete_pass_names = {};
};
}  // namespace fastdeploy
