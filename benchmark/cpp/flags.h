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

#pragma once

#include <unordered_map>
#include "gflags/gflags.h"
#include "fastdeploy/benchmark/utils.h"

#ifdef WIN32
static const char sep = '\\';
#else
static const char sep = '/';
#endif

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(image, "", "Path of the image file.");
DEFINE_string(config_path, "config.txt", "Path of benchmark config.");

static void PrintUsage() {
  std::cout << "Usage: infer_demo --model model_path --image img_path "
               "--config_path config.txt[Path of benchmark config.] "
            << std::endl;
  std::cout << "Default value of device: cpu" << std::endl;
  std::cout << "Default value of backend: default" << std::endl;
  std::cout << "Default value of use_fp16: false" << std::endl;
}

static void PrintBenchmarkInfo(std::unordered_map<std::string,
                               std::string> config_info) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Get model name
  std::vector<std::string> model_names;
  fastdeploy::benchmark::Split(FLAGS_model, model_names, sep);
  if (model_names.empty()) {
    std::cout << "Directory of the inference model is invalid!!!" << std::endl;
    return;
  }
  // Save benchmark info
  std::stringstream ss;
  ss.precision(3);
  ss << "\n======= Model Info =======\n";
  ss << "model_name: " << model_names[model_names.size() - 1] << std::endl;
  ss << "profile_mode: " << config_info["profile_mode"] << std::endl;
  if (config_info["profile_mode"] == "runtime") {
    ss << "include_h2d_d2h: " << config_info["include_h2d_d2h"] << std::endl;
  }
  ss << "\n======= Backend Info =======\n";
  ss << "warmup: " << config_info["warmup"] << std::endl;
  ss << "repeats: " << config_info["repeat"] << std::endl;
  ss << "device: " << config_info["device"] << std::endl;
  if (config_info["device"] == "gpu") {
    ss << "device_id: " << config_info["device_id"] << std::endl;
  }
  ss << "use_fp16: " << config_info["use_fp16"] << std::endl;
  ss << "backend: " << config_info["backend"] << std::endl;
  if (config_info["device"] == "cpu") {
    ss << "cpu_thread_nums: " << config_info["cpu_thread_nums"] << std::endl;
  }
  ss << "collect_memory_info: "
     << config_info["collect_memory_info"] << std::endl;
  if (config_info["collect_memory_info"] == "true") {
    ss << "sampling_interval: " << config_info["sampling_interval"]
       << "ms" << std::endl;
  }
  std::cout << ss.str() << std::endl;
  // Save benchmark info
  fastdeploy::benchmark::ResultManager::SaveBenchmarkResult(ss.str(),
                                        config_info["result_path"]);
#endif
  return;
}
