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

#include "fastdeploy/vision.h"
#include "utils.h"

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
  fastdeploy::vision::DetectionResult res;
  for (int i = 0; i < repeats; i++) {
    if (i % dump_period == 0) {
      DumpCurrentCpuMemoryUsage(cpu_mem_file_name);
      DumpCurrentGpuMemoryUsage(gpu_mem_file_name);
    }
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
  std::cout << "Runtime(ms): " << runtime << "ms." << std::endl;
  std::cout << "End2End(ms): " << end2end << "ms." << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  return true;
}

int main(int argc, char* argv[]) {
  int repeats = 1000;
  int warmup = 200;
  int dump_period = 100;
  std::string cpu_mem_file_name = "result_cpu.txt";
  std::string gpu_mem_file_name = "result_gpu.txt";
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Run model
  if (RunModel(FLAGS_model, FLAGS_image, warmup, repeats, dump_period,
               cpu_mem_file_name, gpu_mem_file_name) != true) {
    exit(1);
  }
  float cpu_mem = GetCpuMemoryUsage(cpu_mem_file_name);
  float gpu_mem = GetGpuMemoryUsage(gpu_mem_file_name);
  std::cout << "cpu_rss_mb: " << cpu_mem << "MB." << std::endl;
  std::cout << "gpu_rss_mb: " << gpu_mem << "MB." << std::endl;
  return 0;
}
