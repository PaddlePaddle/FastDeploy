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
#include <sys/types.h>
#include <dirent.h>
#include <cstring>

#ifdef WIN32
static const char sep = '\\';
#else
static const char sep = '/';
#endif

DEFINE_string(model, "",
              "Required, directory of the inference model."
              "This dir should contains files like: "
              "inference.pdmodel, inference.pdiparams, "
              "inference.yml etc.");
DEFINE_string(image, "",
              "Required, path of the image file.");
DEFINE_string(config_path, "config.txt",
              "Required, path of benchmark config.");
DEFINE_int32(warmup, -1,
             "Optional, number of warmup for profiling, default -1."
             "will force to override the value in config file.");
DEFINE_int32(repeat, -1,
             "Optional, number of repeats for profiling, default -1."
             "will force to override the value in config file.");
DEFINE_int32(xpu_l3_cache, -1,
             "Optional, size xpu l3 cache for profiling, default -1."
              "will force to override the value in config file ");
DEFINE_bool(use_fp16, false,
            "Optional, whether to use fp16, default false."
            "will force to override fp16 option in config file.");
DEFINE_string(model_file, "UNKNOWN",
              "Optional, set specific model file, default 'UNKNOWN'"
              "eg, model.pdmodel, model.onnx, Only support for pure runtime "
              "benchmark bin without pre/post processes.");
DEFINE_string(params_file, "",
              "Optional, set specific params file, default ''"
              "eg, model.pdiparams. Only support for pure runtime "
              "benchmark bin without pre/post processes.");
DEFINE_int32(device_id, -1,
             "Optional, set specific device id for GPU/XPU, default -1."
             "will force to override the value in config file "
             "eg, 0/1/2/...");


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
    if (FLAGS_model_file != "UNKNOWN") {
      model_names.push_back(FLAGS_model_file);
    } else {
      std::cout << "[WARNING] Directory of the inference model is empty!!!"
                << std::endl;
    }
  }
  // Save benchmark info
  int warmup = std::stoi(config_info["warmup"]);
  int repeat = std::stoi(config_info["repeat"]);

  std::stringstream ss;
  ss.precision(3);
  ss << "\n======= Model Info =======\n";
  if (!model_names.empty()) {
    ss << "model_name: " << model_names[model_names.size() - 1] << std::endl;
  }
  ss << "profile_mode: " << config_info["profile_mode"] << std::endl;
  if (config_info["profile_mode"] == "runtime") {
    ss << "include_h2d_d2h: " << config_info["include_h2d_d2h"] << std::endl;
  }
  ss << "\n======= Backend Info =======\n";
  ss << "warmup: " << warmup << std::endl;
  ss << "repeats: " << repeat << std::endl;
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

static bool GetModelResoucesNameFromDir(
  const std::string& path, std::string* resource_name,
  const std::string& suffix = "pdmodel") {
  DIR *p_dir;
  struct dirent *ptr;
  if (!(p_dir = opendir(path.c_str()))) {
    return false;
  }
  bool find = false;
  while ((ptr = readdir(p_dir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      std::string tmp_file_name = ptr->d_name;
      if (tmp_file_name.find(suffix) != std::string::npos) {
        if (suffix == "pdiparams") {
          if (tmp_file_name.find("info") == std::string::npos) {
            find = true;
            *resource_name = tmp_file_name;
            break;
          }
        } else {
          find = true;
          *resource_name = tmp_file_name;
          break;
        }
      } else {
        if (suffix == "yml") {
          if (tmp_file_name.find("yaml") != std::string::npos) {
            find = true;
            *resource_name = tmp_file_name;
            break;
          }
        } else if (suffix == "yaml") {
          if (tmp_file_name.find("yml") != std::string::npos) {
            find = true;
            *resource_name = tmp_file_name;
            break;
          }
        }
      }
    }
  }
  closedir(p_dir);
  return find;
}

static bool UpdateModelResourceName(
  std::string* model_name, std::string* params_name,
  std::string* config_name, fastdeploy::ModelFormat* model_format,
  std::unordered_map<std::string, std::string>& config_info,
  bool use_config_file = true, bool use_quant_model = false) {
  *model_format = fastdeploy::ModelFormat::PADDLE;
  if (!(GetModelResoucesNameFromDir(FLAGS_model, model_name, "pdmodel")
    && GetModelResoucesNameFromDir(FLAGS_model, params_name, "pdiparams"))) {
    std::cout << "Can not find Paddle model resources." << std::endl;
    return false;
  }
  if (use_config_file) {
    if (!GetModelResoucesNameFromDir(FLAGS_model, config_name, "yml")) {
      std::cout << "Can not find config yaml resources." << std::endl;
      return false;
    }
  }

  if (config_info["backend"] == "sophgo") {
    *model_format = fastdeploy::ModelFormat::SOPHGO;
    if (!GetModelResoucesNameFromDir(FLAGS_model, model_name, "bmodel")) {
      std::cout << "Can not find sophgo model resources." << std::endl;
      return false;
    }
  }

  return true;
}
