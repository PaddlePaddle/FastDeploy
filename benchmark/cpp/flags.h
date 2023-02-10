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
#include "fastdeploy/utils/perf.h"

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
DEFINE_int32(dump_period, 100, "How often to collect memory info.");

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

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  if (FLAGS_device == "gpu") {
    option->UseGpu(FLAGS_device_id);
    if (FLAGS_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleInferBackend();
    } else if (FLAGS_backend == "trt" || FLAGS_backend == "paddle_trt") {
      option->UseTrtBackend();
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
    } else if (FLAGS_backend == "lite") {
      option->UsePaddleLiteBackend();
      if (FLAGS_use_fp16) {
        option->EnableLiteFP16();
      }
    } else if (FLAGS_backend == "default") {
      return true;
    } else {
      std::cout << "While inference with CPU, only support "
                   "default/ort/ov/paddle/lite now, "
                << FLAGS_backend << " is not supported." << std::endl;
      return false;
    }
  } else if (FLAGS_device == "xpu") {
    option->UseKunlunXin(FLAGS_device_id);
    if (FLAGS_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleInferBackend();
    } else if (FLAGS_backend == "lite") {
      option->UsePaddleLiteBackend();
      if (FLAGS_use_fp16) {
        option->EnableLiteFP16();
      }
    } else if (FLAGS_backend == "default") {
      return true;
    } else {
      std::cout << "While inference with XPU, only support "
                   "default/ort/paddle/lite now, "
                << FLAGS_backend << " is not supported." << std::endl;
      return false;
    }
  } else {
    std::cerr << "Only support device CPU/GPU/XPU now, " << FLAGS_device
              << " is not supported." << std::endl;
    return false;
  }

  return true;
}
