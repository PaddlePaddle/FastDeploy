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

#include "fastdeploy/benchmark/utils.h"

namespace fastdeploy {
namespace benchmark {

void DumpCurrentCpuMemoryUsage(const std::string& name) {
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
  return;
}

void DumpCurrentGpuMemoryUsage(const std::string& name) {
  std::string command = "nvidia-smi --id=" + std::to_string(FLAGS_device_id) +
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