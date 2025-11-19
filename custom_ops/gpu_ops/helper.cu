// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"
#include <nvml.h>

float bfloat16_to_float(__nv_bfloat16 x) {
  uint32_t tmp_x = *(reinterpret_cast<uint16_t*>(&x));
  tmp_x = tmp_x << 16;
  float float_x = *(reinterpret_cast<float*>(&tmp_x));
  return float_x;
}

template <typename T>
static void PrintMatrix(const T* mat_d,
                        int num,
                        std::string name,
                        int numOfCols) {
  std::vector<T> tmp(num);
  cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

  std::ofstream outfile;
  outfile.open(name + ".dtxt", std::ios::out | std::ios::app);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
        std::is_same<T, int32_t>::value) {
      ss << static_cast<int>(tmp[i]) << " ";
    } else {
      ss << std::setprecision(8) << static_cast<float>(tmp[i]) << " ";
    }
    if (i % numOfCols == numOfCols - 1) {
      ss << std::endl;
    }
  }
  outfile << ss.str();
  outfile.close();
}

GPUMemoryChecker::GPUMemoryChecker() {
  nvmlReturn_t result = nvmlInit_v2();
  if (NVML_SUCCESS != result) {
    throw std::runtime_error("Failed to initialize NVML: " +
                             std::string(nvmlErrorString(result)));
  }

  result = nvmlDeviceGetCount_v2(&deviceCount_);
  if (NVML_SUCCESS != result) {
    nvmlShutdown();
    throw std::runtime_error("Failed to get GPU count: " +
                             std::string(nvmlErrorString(result)));
  }

  getCUDAVisibleDevice();
}

GPUMemoryChecker::~GPUMemoryChecker() { nvmlShutdown(); }

void GPUMemoryChecker::getCUDAVisibleDevice() {
  std::vector<int> devices;
  const char* env_p = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!env_p) {
    for (int i = 0; i < deviceCount_; i++) {
      visible_device_.push_back(i);
      return;
    }
  }

  std::string env_str(env_p);
  std::istringstream stream(env_str);
  std::string device_id;

  while (std::getline(stream, device_id, ',')) {
    visible_device_.push_back(std::stoi(device_id));
    visible_device_mem_usage_.push_back(-1);
  }
  std::cout << "\nVisible NVIDIA GPU devices" << env_str << std::endl;
  return;
}

void GPUMemoryChecker::addCheckPoint(const char* call_file, int call_line) {
  try {
    for (int i = 0; i < visible_device_.size(); i++) {
      unsigned int device_id = visible_device_.at(i);
      nvmlDevice_t device;
      nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(device_id, &device);
      if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get handle for GPU " << device_id << ": "
                  << nvmlErrorString(result) << std::endl;
        continue;
      }

      char name[NVML_DEVICE_NAME_BUFFER_SIZE];
      result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
      if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get name for GPU " << device_id << ": "
                  << nvmlErrorString(result) << std::endl;
        continue;
      }

      nvmlMemory_t memoryInfo;
      result = nvmlDeviceGetMemoryInfo(device, &memoryInfo);
      if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get memory info for GPU " << device_id << ": "
                  << nvmlErrorString(result) << std::endl;
        continue;
      }

      // Check GPU memory
      const char* env_c = std::getenv("MEMCHECKER_CHECK_MEMORY");
      if (env_c) {
        assert(memoryInfo.used <= visible_device_mem_usage_.at(i) &&
               "GPU Memory does not allow growth!");
      }
      visible_device_mem_usage_[i] = memoryInfo.used;
    }

    // Check GPU memory
    const char* env_p = std::getenv("MEMCHECKER_PRINT_MEMORY");
    if (env_p) {
      std::cout << "\nCall Line: " << call_line << "\t";
      for (int i = 0; i < visible_device_.size(); i++) {
        unsigned int device_id = visible_device_.at(i);
        std::cout << "GPU " << device_id << ": "
                  << "  Used memory: "
                  << visible_device_mem_usage_.at(device_id) / (1024 * 1024)
                  << " MB\t";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;

  std::call_once(flag, [&]() {
    int sm_version = GetSMVersion();
    if (sm_version >= 90) {
      enablePDL = getBoolEnv("FD_ENABLE_PDL");
    }
  });
  return enablePDL;
}
