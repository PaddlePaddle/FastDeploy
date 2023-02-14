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

#include <sys/types.h>
#ifdef __linux__
#include <sys/resource.h>
#endif
#include <cmath>

#include "fastdeploy/benchmark/utils.h"

namespace fastdeploy {
namespace benchmark {

std::string Strip(const std::string& str, char ch) {
  int i = 0;
  while (str[i] == ch) {
    i++;
  }
  int j = str.size() - 1;
  while (str[j] == ch) {
    j--;
  }
  return str.substr(i, j + 1 - i);
}

void Split(const std::string& s, std::vector<std::string>& tokens,
           char delim) {
  tokens.clear();
  size_t lastPos = s.find_first_not_of(delim, 0);
  size_t pos = s.find(delim, lastPos);
  while (lastPos != std::string::npos) {
    tokens.emplace_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delim, pos);
    pos = s.find(delim, lastPos);
  }
  return;
}

ResourceUsageMonitor::ResourceUsageMonitor(int sampling_interval_ms, int gpu_id)
    : is_supported_(false),
      sampling_interval_(sampling_interval_ms),
      gpu_id_(gpu_id) {
#ifdef __linux__
  is_supported_ = true;
#else
  is_supported_ = false;
#endif
  if (!is_supported_) {
    FDASSERT(false,
             "Currently ResourceUsageMonitor only supports Linux and ANDROID.")
    return;
  }
}

void ResourceUsageMonitor::Start() {
  if (!is_supported_) {
    return;
  }
  if (check_memory_thd_ != nullptr) {
    FDINFO << "Memory monitoring has already started!" << std::endl;
    return;
  }
  FDINFO << "Start monitoring memory!" << std::endl;
  stop_signal_ = false;
  check_memory_thd_.reset(new std::thread(([this]() {
    // Note we retrieve the memory usage at the very beginning of the thread.
    while (true) {
#ifdef __linux__
      rusage res;
      if (getrusage(RUSAGE_SELF, &res) == 0) {
        max_cpu_mem_ =
            std::max(max_cpu_mem_, static_cast<float>(res.ru_maxrss / 1024.0));
      }
#endif
#if defined(WITH_GPU)
      std::string gpu_mem_info = GetCurrentGpuMemoryInfo(gpu_id_);
      // get max_gpu_mem and max_gpu_util
      std::vector<std::string> gpu_tokens;
      Split(gpu_mem_info, gpu_tokens, ',');
      max_gpu_mem_ = std::max(max_gpu_mem_, stof(gpu_tokens[6]));
      max_gpu_util_ = std::max(max_gpu_util_, stof(gpu_tokens[7]));
#endif
      if (stop_signal_) {
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(sampling_interval_));
    }
  })));
}

void ResourceUsageMonitor::Stop() {
  if (!is_supported_) {
    return;
  }
  if (check_memory_thd_ == nullptr) {
    FDINFO << "Memory monitoring hasn't started yet or has stopped!"
           << std::endl;
    return;
  }
  FDINFO << "Stop monitoring memory!" << std::endl;
  StopInternal();
}

void ResourceUsageMonitor::StopInternal() {
  stop_signal_ = true;
  if (check_memory_thd_ == nullptr) {
    return;
  }
  if (check_memory_thd_ != nullptr) {
    check_memory_thd_->join();
  }
  check_memory_thd_.reset(nullptr);
}

std::string ResourceUsageMonitor::GetCurrentGpuMemoryInfo(int device_id) {
  std::string result = "";
#if defined(__linux__) && defined(WITH_GPU)
  std::string command = "nvidia-smi --id=" + std::to_string(device_id) +
                        " --query-gpu=index,uuid,name,timestamp,memory.total,"
                        "memory.free,memory.used,utilization.gpu,utilization."
                        "memory --format=csv,noheader,nounits";
  FILE* pp = popen(command.data(), "r");
  if (!pp) return "";
  char tmp[1024];

  while (fgets(tmp, sizeof(tmp), pp) != NULL) {
    result += tmp;
  }
  pclose(pp);
#else
  FDASSERT(false,
           "Currently collect gpu memory info only supports Linux in GPU.")
#endif
  return result;
}

}  // namespace benchmark
}  // namespace fastdeploy
