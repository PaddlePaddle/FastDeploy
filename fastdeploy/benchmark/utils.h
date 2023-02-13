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

#include <memory>
#include <thread>  // NOLINT
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace benchmark {
/*! @brief ResourceUsageMonitor object used when to collect memory info.
 */
class FASTDEPLOY_DECL ResourceUsageMonitor {
 public:
   /** \brief  Set sampling_interval_ms and gpu_id for ResourceUsageMonitor.
   *
   * \param[in] sampling_interval_ms How often to collect memory info(ms).
   * \param[in] gpu_id Device(gpu) id, default 0.
   */
  explicit ResourceUsageMonitor(int sampling_interval_ms, int gpu_id = 0);

  ~ResourceUsageMonitor() { StopInternal(); }

  /// Start memory info collect
  void Start();
  /// Stop memory info collect
  void Stop();
  /// Get maximum cpu memory usage
  float GetMaxCpuMem() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_cpu_mem_;
  }
  /// Get maximum gpu memory usage
  float GetMaxGpuMem() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_gpu_mem_;
  }
  /// Get maximum gpu util
  float GetMaxGpuUtil() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_gpu_util_;
  }

  ResourceUsageMonitor(ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor(ResourceUsageMonitor&&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&&) = delete;

 private:
  void StopInternal();
  // Get current gpu memory info
  std::string GetCurrentGpuMemoryInfo(int device_id);

  bool is_supported_ = false;
  bool stop_signal_ = false;
  const int sampling_interval_;
  float max_cpu_mem_ = 0.0f;  // MB
  float max_gpu_mem_ = 0.0f;  // MB
  float max_gpu_util_ = 0.0f;
  const int gpu_id_ = 0;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
};

}  // namespace benchmark
}  // namespace fastdeploy
