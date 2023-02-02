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

#include <sys/types.h>
#include <unistd.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iostream>
#include <string>
#include <vector>

#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision.h"
#include "gflags/gflags.h"

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(image, "", "Path of the image file.");
DEFINE_string(device, "cpu",
              "Type of inference device, support 'cpu' or 'gpu'.");
DEFINE_int32(device_id, 0, "device(gpu) id.");
DEFINE_string(backend, "default",
              "The inference runtime backend, support: ['default', 'ort', "
              "'paddle', 'ov', 'trt', 'paddle_trt']");
DEFINE_bool(
    use_fp16, false,
    "Whether to use FP16 mode, only support 'trt' and 'paddle_trt' backend");

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

// Record current cpu memory usage into file
void DumpCurrentCpuMemoryUsage(std::string name) {
  int iPid = (int)getpid();
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
    option->SetCpuThreadNum(8);
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

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

bool RunModel(std::string model_file, std::string image_file, size_t warmup,
              size_t repeats, size_t dump_period, std::string cpu_mem_file_name,
              std::string gpu_mem_file_name) {
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return false;
  }
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return false;
  }
  auto im = cv::imread(image_file);
  // Step1: warm up for warmup times
  std::cout << "Warmup " << warmup << " times..." << std::endl;
  for (int i = 0; i < warmup; i++) {
    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return false;
    }
  }

  // wait for 2 second to ensure that memory gets stable
  sleep(2);
  DumpCurrentCpuMemoryUsage(cpu_mem_file_name);
  DumpCurrentGpuMemoryUsage(gpu_mem_file_name);
  std::vector<float> end2end_statis;
  // Step2: repeat for repeats times
  model.EnableRecordTimeOfRuntime();
  std::cout << "Counting time..." << std::endl;
  fastdeploy::TimeCounter tc;
  for (int i = 0; i < repeats; i++) {
    if (i % dump_period == 0) {
      DumpCurrentCpuMemoryUsage(cpu_mem_file_name);
      DumpCurrentGpuMemoryUsage(gpu_mem_file_name);
    }
    fastdeploy::vision::DetectionResult res;
    tc.Start();
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return false;
    }
    tc.End();
    end2end_statis.push_back(tc.Duration() * 1000);
  }
  float end2end = std::accumulate(end2end_statis.end() - repeats,
                                  end2end_statis.end(), 0.f) /
                  repeats;
  auto runtime_statis = model.PrintStatisInfoOfRuntime();
  float runtime = runtime_statis["avg_time"] * 1000;
  float backend_time = runtime_statis["backend_avg_time"] * 1000;
  std::cout << "Backend Runtime(ms): " << backend_time << "ms." << std::endl;
  std::cout << "Runtime(ms): " << runtime << "ms." << std::endl;
  std::cout << "End2End(ms): " << end2end << "ms." << std::endl;
  return true;
}

int main(int argc, char* argv[]) {
  int repeats = 100;
  int warmup = 20;
  int dump_period = 10;
  std::string cpu_mem_file_name = "result_cpu.txt";
  std::string gpu_mem_file_name = "result_gpu.txt";
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Run model and check memory leak.
  if (RunModel(FLAGS_model, FLAGS_image, warmup, repeats, dump_period,
               cpu_mem_file_name, gpu_mem_file_name) != true) {
    exit(1);
  }
  std::cout << "Test end, heap result is stored into " << cpu_mem_file_name
            << "\n";
  std::cout << "Test end, heap result is stored into " << gpu_mem_file_name
            << "\n";
  return 0;
}
