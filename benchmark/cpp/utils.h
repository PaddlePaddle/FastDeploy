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

#ifndef BENCHMARK_CPP_UTILS_H_
#define BENCHMARK_CPP_UTILS_H_
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "fastdeploy/utils/perf.h"

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(image, "", "Path of the image file.");
DEFINE_string(device, "cpu",
              "Type of inference device, support 'cpu' or 'gpu'.");
DEFINE_int32(device_id, 0, "device(gpu) id.");
DEFINE_int32(warmup, 200, "Number of warmup for profiling.");
DEFINE_int32(repeat, 1000, "Number of repeats for profiling.");
DEFINE_string(profile_mode, "runtime", "runtime or end2end.");
DEFINE_string(backend, "default",
              "The inference runtime backend, support: ['default', 'ort', "
              "'paddle', 'ov', 'trt', 'paddle_trt']");
DEFINE_int32(cpu_thread_nums, 8, "Set numbers of cpu thread.");
DEFINE_bool(
    include_h2d_d2h, false, "Whether run profiling with h2d and d2h.");
DEFINE_bool(
    use_fp16, false,
    "Whether to use FP16 mode, only support 'trt' and 'paddle_trt' backend");
DEFINE_bool(
    collect_memory_info, false, "Whether to collect memory info");
DEFINE_int32(dump_period, 100, "How often to collect memory info.");

void PrintUsage() {
  std::cout << "Usage: infer_demo --model model_path --image img_path --device "
               "[cpu|gpu] --backend "
               "[default|ort|paddle|ov|trt|paddle_trt] "
               "--use_fp16 false"
            << std::endl;
  std::cout << "Default value of device: cpu" << std::endl;
  std::cout << "Default value of backend: default" << std::endl;
  std::cout << "Default value of use_fp16: false" << std::endl;
}

std::string strip(const std::string& str, char ch = ' ') {
  // Remove the ch characters at both ends of str
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

// Record current cpu memory usage into file
void DumpCurrentCpuMemoryUsage(std::string name) {
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

// Record current gpu memory usage into file
void DumpCurrentGpuMemoryUsage(std::string name) {
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

// Get Max cpu memory usage
float GetCpuMemoryUsage(std::string name) {
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

// Get Max gpu memory usage
float GetGpuMemoryUsage(std::string name) {
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

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  if (FLAGS_device == "gpu") {
    option->UseGpu();
    if (FLAGS_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleInferBackend();
    } else if (FLAGS_backend == "trt" || FLAGS_backend == "paddle_trt") {
      option->UseTrtBackend();
      option->SetTrtInputShape("input", {1, 3, 112, 112});
      if (FLAGS_backend == "paddle_trt") {
        option->EnablePaddleToTrt();
      }
      if (FLAGS_use_fp16) {
        option->EnableTrtFP16();
      }
    } else if (FLAGS_backend == "default") {
      return true;
    } else {
      std::cout << "While inference with GPU, only support "
                   "default/ort/paddle/trt/paddle_trt now, "
                << FLAGS_backend << " is not supported." << std::endl;
      return false;
    }
  } else if (FLAGS_device == "cpu") {
    option->SetCpuThreadNum(FLAGS_cpu_thread_nums);
    if (FLAGS_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "ov") {
      option->UseOpenVINOBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleInferBackend();
    } else if (FLAGS_backend == "default") {
      return true;
    } else {
      std::cout << "While inference with CPU, only support "
                   "default/ort/ov/paddle now, "
                << FLAGS_backend << " is not supported." << std::endl;
      return false;
    }
  } else {
    std::cerr << "Only support device CPU/GPU now, " << FLAGS_device
              << " is not supported." << std::endl;
    return false;
  }

  return true;
}

#endif  // BENCHMARK_CPP_UTILS_H_
