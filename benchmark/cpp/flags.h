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

#include "gflags/gflags.h"
#include "fastdeploy/benchmark/utils.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(image, "", "Path of the image file.");
DEFINE_string(device, "cpu",
              "Type of inference device, support 'cpu/gpu/xpu'.");
DEFINE_int32(device_id, 0, "device(gpu/xpu/...) id.");
DEFINE_int32(warmup, 200, "Number of warmup for profiling.");
DEFINE_int32(repeat, 1000, "Number of repeats for profiling.");
DEFINE_string(profile_mode, "runtime", "runtime or end2end.");
DEFINE_string(backend, "default",
              "The inference runtime backend, support: ['default', 'ort', "
              "'paddle', 'ov', 'trt', 'paddle_trt', 'lite']");
DEFINE_int32(cpu_thread_nums, 8, "Set numbers of cpu thread.");
DEFINE_bool(
    include_h2d_d2h, false, "Whether run profiling with h2d and d2h.");
DEFINE_bool(
    use_fp16, false,
    "Whether to use FP16 mode, only support 'trt', 'paddle_trt' "
    "and 'lite' backend");
DEFINE_bool(
    collect_memory_info, false, "Whether to collect memory info");
DEFINE_int32(sampling_interval, 50, "How often to collect memory info(ms).");

void PrintUsage() {
  std::cout << "Usage: infer_demo --model model_path --image img_path --device "
               "[cpu|gpu|xpu] --backend "
               "[default|ort|paddle|ov|trt|paddle_trt|lite] "
               "--use_fp16 false"
            << std::endl;
  std::cout << "Default value of device: cpu" << std::endl;
  std::cout << "Default value of backend: default" << std::endl;
  std::cout << "Default value of use_fp16: false" << std::endl;
}

void PrintBenchmarkInfo() {
  // Get model name
  std::vector<std::string> model_names;
  fastdeploy::benchmark::Split(FLAGS_model, model_names, sep);
  // Save benchmark info
  std::stringstream ss;
  ss.precision(3);
  ss << "\n======= Model Info =======\n";
  ss << "model_name: " << model_names[model_names.size() - 1] << std::endl;
  ss << "profile_mode: " << FLAGS_profile_mode << std::endl;
  if (FLAGS_profile_mode == "runtime") {
    ss << "include_h2d_d2h: " << FLAGS_include_h2d_d2h << std::endl;
  }
  ss << "\n======= Backend Info =======\n";
  ss << "warmup: " << FLAGS_warmup << std::endl;
  ss << "repeats: " << FLAGS_repeat << std::endl;
  ss << "device: " << FLAGS_device << std::endl;
  if (FLAGS_device == "gpu") {
    ss << "device_id: " << FLAGS_device_id << std::endl;
  }
  ss << "backend: " << FLAGS_backend << std::endl;
  if (FLAGS_device == "cpu") {
    ss << "cpu_thread_nums: " << FLAGS_cpu_thread_nums << std::endl;
  }
  ss << "use_fp16: " << FLAGS_use_fp16 << std::endl;
  ss << "collect_memory_info: " << FLAGS_collect_memory_info << std::endl;
  if (FLAGS_collect_memory_info) {
    ss << "sampling_interval: " << std::to_string(FLAGS_sampling_interval)
       << "ms" << std::endl;
  }
  std::cout << ss.str() << std::endl;
  return;
}
