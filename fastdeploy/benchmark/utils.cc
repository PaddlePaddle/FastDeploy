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
#include "fastdeploy/utils/path.h"
#include "fastdeploy/vision/utils/utils.h"

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

void Split(const std::string& s, std::vector<std::string>& tokens, char delim) {
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

/// Utils for precision evaluation
std::vector<std::string> ReadLines(const std::string& path) {
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
  fin.close();
  return lines;
}

std::map<std::string, std::vector<std::string>> SplitDataLine(
    const std::string& data_line) {
  std::map<std::string, std::vector<std::string>> dict;
  std::vector<std::string> tokens, value_tokens;
  Split(data_line, tokens, ':');
  std::string key = tokens[0];
  std::string value = tokens[1];
  Split(value, value_tokens, ',');
  dict[key] = value_tokens;
  return dict;
}

bool ResultManager::SaveFDTensor(const FDTensor&& tensor,
                                 const std::string& path) {
  if (tensor.CpuData() == nullptr || tensor.Numel() <= 0) {
    FDERROR << "Input tensor is empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(15);
  if (tensor.Dtype() != FDDataType::FP32 &&
      tensor.Dtype() != FDDataType::INT32 &&
      tensor.Dtype() != FDDataType::INT64) {
    FDERROR << "Only support FP32/INT32/INT64 now, but got "
            << Str(tensor.dtype) << std::endl;
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
  const void* data_ptr = tensor.CpuData();
  for (int i = 0; i < tensor.Numel(); ++i) {
    if (tensor.Dtype() == FDDataType::INT64) {
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const int64_t*>(data_ptr))[i] << ",";
      } else {
        fs << (static_cast<const int64_t*>(data_ptr))[i];
      }
    } else if (tensor.Dtype() == FDDataType::INT32) {
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const int32_t*>(data_ptr))[i] << ",";
      } else {
        fs << (static_cast<const int32_t*>(data_ptr))[i];
      }
    } else {  // FP32
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const float*>(data_ptr))[i] << ",";
      } else {
        fs << (static_cast<const float*>(data_ptr))[i];
      }
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
  std::map<std::string, std::vector<std::string>> data;
  // name
  data = SplitDataLine(lines[0]);
  tensor->name = data.begin()->first;
  // shape
  data = SplitDataLine(lines[1]);
  tensor->shape.clear();
  for (const auto& s : data.begin()->second) {
    tensor->shape.push_back(std::stol(s));
  }
  // dtype
  data = SplitDataLine(lines[2]);
  if (data.begin()->second.at(0) == Str(FDDataType::INT64)) {
    tensor->dtype = FDDataType::INT64;
  } else if (data.begin()->second.at(0) == Str(FDDataType::INT32)) {
    tensor->dtype = FDDataType::INT32;
  } else if (data.begin()->second.at(0) == Str(FDDataType::FP32)) {
    tensor->dtype = FDDataType::FP32;
  } else {
    FDERROR << "Only support FP32/INT64/INT32 now, but got "
            << data.begin()->second.at(0) << std::endl;
    return false;
  }
  // data
  data = SplitDataLine(lines[3]);
  tensor->Allocate(tensor->shape, tensor->dtype, tensor->name);
  if (tensor->dtype == FDDataType::INT64) {
    int64_t* mutable_data_ptr = static_cast<int64_t*>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stol(data.begin()->second[i]);
    }
  } else if (tensor->dtype == FDDataType::INT32) {
    int32_t* mutable_data_ptr = static_cast<int32_t*>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stoi(data.begin()->second[i]);
    }
  } else {  // FP32
    float* mutable_data_ptr = static_cast<float*>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stof(data.begin()->second[i]);
    }
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
  fs.precision(15);
  // boxes
  fs << "boxes:";
  for (int i = 0; i < res.boxes.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if ((i == res.boxes.size() - 1) && (j == 3)) {
        fs << res.boxes[i][j];
      } else {
        fs << res.boxes[i][j] << ",";
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
  // TODO(qiuyanjun): dump masks
  fs.close();
  return true;
}

bool ResultManager::LoadDetectionResult(vision::DetectionResult* res,
                                        const std::string& path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from" << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::map<std::string, std::vector<std::string>> data;

  // boxes
  data = SplitDataLine(lines[0]);
  int boxes_num = data.begin()->second.size() / 4;
  res->Resize(boxes_num);
  for (int i = 0; i < boxes_num; ++i) {
    res->boxes[i][0] = std::stof(data.begin()->second[i * 4 + 0]);
    res->boxes[i][1] = std::stof(data.begin()->second[i * 4 + 1]);
    res->boxes[i][2] = std::stof(data.begin()->second[i * 4 + 2]);
    res->boxes[i][3] = std::stof(data.begin()->second[i * 4 + 3]);
  }
  // scores
  data = SplitDataLine(lines[1]);
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->scores[i] = std::stof(data.begin()->second[i]);
  }
  // label_ids
  data = SplitDataLine(lines[2]);
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->label_ids[i] = std::stoi(data.begin()->second[i]);
  }
  // TODO(qiuyanjun): load masks
  return true;
}

TensorDiff ResultManager::CalculateDiffStatis(FDTensor* lhs, FDTensor* rhs) {
  if (lhs->Numel() != rhs->Numel() || lhs->Dtype() != rhs->Dtype()) {
    FDASSERT(false, "The size and dtype of input FDTensor must be equal!")
  }
  FDDataType dtype = lhs->Dtype();
  int numel = lhs->Numel();
  if (dtype != FDDataType::FP32 && dtype != FDDataType::INT64 &&
      dtype != FDDataType::INT32) {
    FDASSERT(false, "Only support FP32/INT64/INT32 now, but got %s",
             Str(dtype).c_str())
  }
  if (dtype == FDDataType::INT64) {
    std::vector<int64_t> tensor_diff(numel);
    const int64_t* lhs_data_ptr = static_cast<const int64_t*>(lhs->CpuData());
    const int64_t* rhs_data_ptr = static_cast<const int64_t*>(rhs->CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<int64_t>(tensor_diff.data(), numel, &(diff.mean),
                                 &(diff.max), &(diff.min));
    return diff;
  } else if (dtype == FDDataType::INT32) {
    std::vector<int32_t> tensor_diff(numel);
    const int32_t* lhs_data_ptr = static_cast<const int32_t*>(lhs->CpuData());
    const int32_t* rhs_data_ptr = static_cast<const int32_t*>(rhs->CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<float>(tensor_diff.data(), numel, &(diff.mean),
                               &(diff.max), &(diff.min));
    return diff;
  } else {  // FP32
    std::vector<float> tensor_diff(numel);
    const float* lhs_data_ptr = static_cast<const float*>(lhs->CpuData());
    const float* rhs_data_ptr = static_cast<const float*>(rhs->CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<float>(tensor_diff.data(), numel, &(diff.mean),
                               &(diff.max), &(diff.min));
    return diff;
  }
}

DetectionDiff ResultManager::CalculateDiffStatis(vision::DetectionResult* lhs,
                                                 vision::DetectionResult* rhs,
                                                 float score_threshold) {
  // lex sort by x(w) & y(h)
  vision::utils::LexSortDetectionResultByXY(lhs);
  vision::utils::LexSortDetectionResultByXY(rhs);
  // get value diff & trunc it by score_threshold
  const int boxes_num = std::min(lhs->boxes.size(), rhs->boxes.size());
  std::vector<float> boxes_diff;
  std::vector<float> scores_diff;
  std::vector<int32_t> labels_diff;
  // TODO(qiuyanjun): process the diff of masks.
  for (int i = 0; i < boxes_num; ++i) {
    if (lhs->scores[i] > score_threshold && rhs->scores[i] > score_threshold) {
      scores_diff.push_back(lhs->scores[i] - rhs->scores[i]);
      labels_diff.push_back(lhs->label_ids[i] - rhs->label_ids[i]);
      boxes_diff.push_back(lhs->boxes[i][0] - rhs->boxes[i][0]);
      boxes_diff.push_back(lhs->boxes[i][1] - rhs->boxes[i][1]);
      boxes_diff.push_back(lhs->boxes[i][2] - rhs->boxes[i][2]);
      boxes_diff.push_back(lhs->boxes[i][3] - rhs->boxes[i][3]);
    }
  }
  FDASSERT(boxes_diff.size() > 0,
           "Can't get any valid boxes while score_threshold is %f, "
           "The boxes.size of lhs is %d, the boxes.size of rhs is %d",
           score_threshold, lhs->boxes.size(), rhs->boxes.size())

  DetectionDiff diff;
  CalculateStatisInfo<float>(boxes_diff.data(), boxes_diff.size(),
                             &(diff.boxes.mean), &(diff.boxes.max),
                             &(diff.boxes.min));
  CalculateStatisInfo<float>(scores_diff.data(), scores_diff.size(),
                             &(diff.scores.mean), &(diff.scores.max),
                             &(diff.scores.min));
  CalculateStatisInfo<int32_t>(labels_diff.data(), labels_diff.size(),
                               &(diff.labels.mean), &(diff.labels.max),
                               &(diff.labels.min));
  diff.mean = diff.boxes.mean;
  diff.max = diff.boxes.max;
  diff.min = diff.boxes.min;
  return diff;
}

}  // namespace benchmark
}  // namespace fastdeploy
