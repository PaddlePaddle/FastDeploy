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
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif
#include <cmath>

#include "fastdeploy/benchmark/utils.h"

namespace fastdeploy {
namespace benchmark {

// Remove the ch characters at both ends of str
static std::string strip(const std::string& str, char ch = ' ') {
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

void DumpCurrentCpuMemoryUsage(const std::string& name) {
#if defined(__linux__) || defined(__ANDROID__)
  int iPid = static_cast<int>(getpid());
  std::string command = "pmap -x " + std::to_string(iPid) + " | grep total";
  FILE* pp = popen(command.data(), "r");
  if (!pp) return;
  char tmp[1024];

  while (fgets(tmp, sizeof(tmp), pp) != NULL) {
    std::ofstream write;
    write.open(name, std::ios::app);
    write << tmp;
    write.close();
  }
  pclose(pp);
#else
  FDASSERT(false,
           "Currently collect cpu memory info only supports Linux and ANDROID.")
#endif
  return;
}

void DumpCurrentGpuMemoryUsage(const std::string& name, int device_id) {
#if defined(__linux__) && defined(WITH_GPU)
  std::string command = "nvidia-smi --id=" + std::to_string(device_id) +
                        " --query-gpu=index,uuid,name,timestamp,memory.total,"
                        "memory.free,memory.used,utilization.gpu,utilization."
                        "memory --format=csv,noheader,nounits";
  FILE* pp = popen(command.data(), "r");
  if (!pp) return;
  char tmp[1024];

  while (fgets(tmp, sizeof(tmp), pp) != NULL) {
    std::ofstream write;
    write.open(name, std::ios::app);
    write << tmp;
    write.close();
  }
  pclose(pp);
#else
  FDASSERT(false,
           "Currently collect gpu memory info only supports Linux in GPU.")
#endif
  return;
}

float GetCpuMemoryUsage(const std::string& name) {
  std::ifstream read(name);
  std::string line;
  float max_cpu_mem = -1;
  while (getline(read, line)) {
    std::stringstream ss(line);
    std::string tmp;
    std::vector<std::string> nums;
    while (getline(ss, tmp, ' ')) {
      tmp = strip(tmp);
      if (tmp.empty()) continue;
      nums.push_back(tmp);
    }
    max_cpu_mem = std::max(max_cpu_mem, stof(nums[3]));
  }
  return max_cpu_mem / 1024;
}

float GetGpuMemoryUsage(const std::string& name) {
  std::ifstream read(name);
  std::string line;
  float max_gpu_mem = -1;
  while (getline(read, line)) {
    std::stringstream ss(line);
    std::string tmp;
    std::vector<std::string> nums;
    while (getline(ss, tmp, ',')) {
      tmp = strip(tmp);
      if (tmp.empty()) continue;
      nums.push_back(tmp);
    }
    max_gpu_mem = std::max(max_gpu_mem, stof(nums[6]));
  }
  return max_gpu_mem;
}

}  // namespace benchmark
}  // namespace fastdeploy
