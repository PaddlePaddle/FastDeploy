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

#include "fastdeploy/benchmark/utils.h"
#include "fastdeploy/utils/perf.h"

#define BENCHMARK_MODEL(MODEL_NAME, BENCHMARK_FUNC)                         \
{                                                                           \
  std::cout << "====" << #MODEL_NAME << "====" << std::endl;                \
  if (!MODEL_NAME.Initialized()) {                                          \
    std::cerr << "Failed to initialize." << std::endl;                      \
    return 0;                                                               \
  }                                                                         \
  auto __im__ = cv::imread(FLAGS_image);                                    \
  fastdeploy::benchmark::ResourceUsageMonitor __resource_moniter__(         \
      FLAGS_sampling_interval, FLAGS_device_id);                            \
  if (FLAGS_collect_memory_info) {                                          \
    __resource_moniter__.Start();                                           \
  }                                                                         \
  if (FLAGS_profile_mode == "runtime") {                                    \
    if (!BENCHMARK_FUNC) {                                                  \
      std::cerr << "Failed to predict." << std::endl;                       \
      return 0;                                                             \
    }                                                                       \
    double __profile_time__ = MODEL_NAME.GetProfileTime() * 1000;           \
    std::cout << "Runtime(ms): " << __profile_time__ << "ms." << std::endl; \
  } else {                                                                  \
    std::cout << "Warmup " << FLAGS_warmup << " times..." << std::endl;     \
    for (int __i__ = 0; __i__ < FLAGS_warmup; __i__++) {                    \
      if (!BENCHMARK_FUNC) {                                                \
        std::cerr << "Failed to predict." << std::endl;                     \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    std::cout << "Counting time..." << std::endl;                           \
    std::cout << "Repeat " << FLAGS_repeat << " times..." << std::endl;     \
    fastdeploy::TimeCounter __tc__;                                         \
    __tc__.Start();                                                         \
    for (int __i__ = 0; __i__ < FLAGS_repeat; __i__++) {                    \
      if (!BENCHMARK_FUNC) {                                                \
        std::cerr << "Failed to predict." << std::endl;                     \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    __tc__.End();                                                           \
    double __end2end__ = __tc__.Duration() / FLAGS_repeat * 1000;           \
    std::cout << "End2End(ms): " << __end2end__ << "ms." << std::endl;      \
  }                                                                         \
  if (FLAGS_collect_memory_info) {                                          \
    float __cpu_mem__ = __resource_moniter__.GetMaxCpuMem();                \
    float __gpu_mem__ = __resource_moniter__.GetMaxGpuMem();                \
    float __gpu_util__ = __resource_moniter__.GetMaxGpuUtil();              \
    std::cout << "cpu_pss_mb: " << __cpu_mem__ << "MB." << std::endl;       \
    std::cout << "gpu_pss_mb: " << __gpu_mem__ << "MB." << std::endl;       \
    std::cout << "gpu_util: " << __gpu_util__ << std::endl;                 \
    __resource_moniter__.Stop();                                            \
  }                                                                         \
}
