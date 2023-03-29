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
  if (!MODEL_NAME.Initialized()) {                                          \
    std::cerr << "Failed to initialize." << std::endl;                      \
    return 0;                                                               \
  }                                                                         \
  std::unordered_map<std::string, std::string> __config_info__;             \
  fastdeploy::benchmark::ResultManager::LoadBenchmarkConfig(                \
                             FLAGS_config_path, &__config_info__);          \
  std::stringstream __ss__;                                                 \
  __ss__.precision(6);                                                      \
  fastdeploy::benchmark::ResourceUsageMonitor __resource_moniter__(         \
                     std::stoi(__config_info__["sampling_interval"]),       \
                     std::stoi(__config_info__["device_id"]));              \
  if (__config_info__["collect_memory_info"] == "true") {                   \
    __resource_moniter__.Start();                                           \
  }                                                                         \
  if (__config_info__["profile_mode"] == "runtime") {                       \
    if (!BENCHMARK_FUNC) {                                                  \
      std::cerr << "Failed to predict." << std::endl;                       \
      __ss__ << "Runtime(ms): Failed" << std::endl;                         \
      if (__config_info__["collect_memory_info"] == "true") {               \
        __ss__ << "cpu_rss_mb: Failed" << std::endl;                        \
        __ss__ << "gpu_rss_mb: Failed" << std::endl;                        \
        __ss__ << "gpu_util: Failed" << std::endl;                          \
      }                                                                     \
      fastdeploy::benchmark::ResultManager::SaveBenchmarkResult(            \
                          __ss__.str(), __config_info__["result_path"]);    \
      return 0;                                                             \
    }                                                                       \
    double __profile_time__ = MODEL_NAME.GetProfileTime() * 1000;           \
    std::cout << "Runtime(ms): " << __profile_time__ << "ms." << std::endl; \
    __ss__ << "Runtime(ms): " << __profile_time__ << "ms." << std::endl;    \
  } else {                                                                  \
    std::cout << "Warmup "                                                  \
              << __config_info__["warmup"]                                  \
              << " times..." << std::endl;                                  \
    int __warmup__ = std::stoi(__config_info__["warmup"]);                  \
    for (int __i__ = 0; __i__ < __warmup__; __i__++) {                      \
      if (!BENCHMARK_FUNC) {                                                \
        std::cerr << "Failed to predict." << std::endl;                     \
        __ss__ << "End2End(ms): Failed" << std::endl;                       \
        if (__config_info__["collect_memory_info"] == "true") {             \
          __ss__ << "cpu_rss_mb: Failed" << std::endl;                      \
          __ss__ << "gpu_rss_mb: Failed" << std::endl;                      \
          __ss__ << "gpu_util: Failed" << std::endl;                        \
        }                                                                   \
        fastdeploy::benchmark::ResultManager::SaveBenchmarkResult(          \
                          __ss__.str(), __config_info__["result_path"]);    \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    std::cout << "Counting time..." << std::endl;                           \
    std::cout << "Repeat "                                                  \
              << __config_info__["repeat"]                                  \
              << " times..." << std::endl;                                  \
    fastdeploy::TimeCounter __tc__;                                         \
    __tc__.Start();                                                         \
    int __repeat__ = std::stoi(__config_info__["repeat"]);                  \
    for (int __i__ = 0; __i__ < __repeat__; __i__++) {                      \
      if (!BENCHMARK_FUNC) {                                                \
        std::cerr << "Failed to predict." << std::endl;                     \
        __ss__ << "End2End(ms): Failed" << std::endl;                       \
        if (__config_info__["collect_memory_info"] == "true") {             \
          __ss__ << "cpu_rss_mb: Failed" << std::endl;                      \
          __ss__ << "gpu_rss_mb: Failed" << std::endl;                      \
          __ss__ << "gpu_util: Failed" << std::endl;                        \
        }                                                                   \
        fastdeploy::benchmark::ResultManager::SaveBenchmarkResult(          \
                          __ss__.str(), __config_info__["result_path"]);    \
        return 0;                                                           \
      }                                                                     \
    }                                                                       \
    __tc__.End();                                                           \
    double __end2end__ = __tc__.Duration() / __repeat__ * 1000;             \
    std::cout << "End2End(ms): " << __end2end__ << "ms." << std::endl;      \
    __ss__ << "End2End(ms): " << __end2end__ << "ms." << std::endl;         \
  }                                                                         \
  if (__config_info__["collect_memory_info"] == "true") {                   \
    float __cpu_mem__ = __resource_moniter__.GetMaxCpuMem();                \
    float __gpu_mem__ = __resource_moniter__.GetMaxGpuMem();                \
    float __gpu_util__ = __resource_moniter__.GetMaxGpuUtil();              \
    std::cout << "cpu_rss_mb: " << __cpu_mem__ << "MB." << std::endl;       \
    __ss__ << "cpu_rss_mb: " << __cpu_mem__ << "MB." << std::endl;          \
    std::cout << "gpu_rss_mb: " << __gpu_mem__ << "MB." << std::endl;       \
    __ss__ << "gpu_rss_mb: " << __gpu_mem__ << "MB." << std::endl;          \
    std::cout << "gpu_util: " << __gpu_util__ << std::endl;                 \
    __ss__ << "gpu_util: " << __gpu_util__ << "MB." << std::endl;           \
    __resource_moniter__.Stop();                                            \
  }                                                                         \
  fastdeploy::benchmark::ResultManager::SaveBenchmarkResult(__ss__.str(),   \
                                         __config_info__["result_path"]);   \
}
