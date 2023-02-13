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
#include "fastdeploy/utils/path.h"

namespace fastdeploy {
namespace benchmark {

// Remove the ch characters at both ends of str
static std::string Strip(const std::string& str, char ch = ' ') {
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

// Split string
static void Split(const std::string& s, std::vector<std::string>& tokens,
                  char delim = ' ') {
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
#if defined(__linux__) || defined(__ANDROID__)
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
      std::string cpu_mem_info = GetCurrentCpuMemoryInfo();
      // get max_cpu_mem
      std::vector<std::string> cpu_tokens;
      Split(cpu_mem_info, cpu_tokens, ' ');
      max_cpu_mem_ = std::max(max_cpu_mem_, stof(cpu_tokens[3]) / 1024);
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

std::string ResourceUsageMonitor::GetCurrentCpuMemoryInfo() {
  std::string result = "";
#if defined(__linux__) || defined(__ANDROID__)
  int iPid = static_cast<int>(getpid());
  std::string command = "pmap -x " + std::to_string(iPid) + " | grep total";
  FILE* pp = popen(command.data(), "r");
  if (!pp) return "";
  char tmp[1024];

  while (fgets(tmp, sizeof(tmp), pp) != NULL) {
    result += tmp;
  }
  pclose(pp);
#else
  FDASSERT(false,
           "Currently collect cpu memory info only supports Linux and ANDROID.")
#endif
  return result;
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

/// Diff values for precision evaluation
bool TensorDiff::IsHasDiff() {
  // TODO(qiuyanjun) Reimplement this func
  return has_diff && status;
}

bool DetectionDiff::IsHasDiff() {
  // TODO(qiuyanjun) Reimplement this func
  return has_diff && status;
}

/// Utils for precision evaluation
static std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream fin(path);
  std::vector<std::string> lines;
  std::string line;
  if (fin.is_open()) {
    while (getline(fin, line)) {
      lines.push_back(line);
    }
  } else {
    FDERROR << "Failed to open file " << path << std::endl;
    std::abort();
  }
  return lines;
}

bool ResultManager::SaveFDTensor(const FDTensor&& tensor,
                                 const std::string& path) {
  if (tensor.CpuData() == nullptr || tensor.Numel() <= 0) {
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  if (tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support FP32 now, but got " << Str(tensor.dtype)
            << std::endl;
    return false;
  }
  // name
  fs << "name:" << tensor.name << "\n";
  // shape
  fs << "shape:";
  for (int i = 0; i < tensor.shape.size(); ++i) {
    if (i < tensor.shape.size() - 1) {
      fs << i << ",";
    } else {
      fs << i;
    }
  }
  fs << "\n";
  // dtype
  fs << "dtype:" << Str(tensor.dtype) << "\n";
  // data
  const float* data_ptr = static_cast<const float*>(tensor.CpuData());
  for (int i = 0; i < tensor.Numel(); ++i) {
    if (i < tensor.Numel() - 1) {
      fs << data_ptr[i] << ",";
    } else {
      fs << data_ptr[i];
    }
  }
  fs << "\n";
  fs.close();

  return true;
}

bool ResultManager::LoadFDTensor(FDTensor* tensor, const std::string& path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from" << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::vector<std::string> tokens;
  // name
  Split(lines[0], tokens, ':');
  tensor->name = tokens[0];
  // shape
  Split(lines[1], tokens, ':');
  std::vector<std::string> shape_tokens;
  Split(tokens[1], shape_tokens, ',');
  tensor->shape.clear();
  for (const auto& s : shape_tokens) {
    tensor->shape.push_back(std::stol(s));
  }
  // dtype
  Split(lines[2], tokens, ':');
  if (tokens[1] != Str(FDDataType::FP32)) {
    FDERROR << "Only support FP32 now, but got " << Str(FDDataType::FP32)
            << std::endl;
    return false;
  }
  tensor->dtype = FDDataType::FP32;
  // data
  Split(lines[3], tokens, ':');
  std::vector<std::string> data_tokens;
  Split(tokens[1], data_tokens, ',');
  tensor->Allocate(tensor->shape, tensor->dtype, tensor->name);
  float* mutable_data_ptr = static_cast<float*>(tensor->MutableData());
  for (int i = 0; i < data_tokens.size(); ++i) {
    mutable_data_ptr[i] = std::stof(data_tokens[i]);
  }

  return true;
}

bool ResultManager::SaveDetectionResult(const vision::DetectionResult& res,
                                        const std::string& path) {
  if (res.boxes.empty()) {
    FDERROR << "DetectionResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  // boxes
  fs << "boxes:";
  for (int i = 0; i < res.boxes.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if ((i < res.boxes.size() - 1) && (j < 3)) {
        fs << res.boxes[i][j] << ",";
      } else {
        fs << res.boxes[i][j];
      }
    }
  }
  fs << "\n";
  // scores
  fs << "scores:";
  for (int i = 0; i < res.scores.size(); ++i) {
    if (i < res.scores.size() - 1) {
      fs << res.scores[i] << ",";
    } else {
      fs << res.scores[i];
    }
  }
  fs << "\n";
  // label_ids
  fs << "label_ids:";
  for (int i = 0; i < res.label_ids.size(); ++i) {
    if (i < res.label_ids.size() - 1) {
      fs << res.label_ids[i] << ",";
    } else {
      fs << res.label_ids[i];
    }
  }
  fs << "\n";
  fs.close();

  return true;
}

bool ResultManager::LoadDetectionResult(vision::DetectionResult* res,
                                        const std::string& path) {
  return false;
}

TensorDiff ResultManager::CalcDiffFrom(const FDTensor& lhs,
                                       const FDTensor& rhs) {
  TensorDiff diff;
  return diff;
}

DetectionDiff ResultManager::CalcDiffFrom(const vision::DetectionResult& lhs,
                                          const vision::DetectionResult& rhs) {
  DetectionDiff diff;
  return diff;
}

}  // namespace benchmark
}  // namespace fastdeploy
