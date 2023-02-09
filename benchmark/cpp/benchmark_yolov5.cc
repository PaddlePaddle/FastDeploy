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

#include "fastdeploy/benchmark/utils.h"
#include "fastdeploy/vision.h"
#include "flags.h"

bool RunModel(std::string model_file, std::string image_file, size_t warmup,
              size_t repeats, size_t sampling_interval) {
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return false;
  }
  if (FLAGS_profile_mode == "runtime") {
    option.EnableProfiling(FLAGS_include_h2d_d2h, repeats, warmup);
  }
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return false;
  }
  auto im = cv::imread(image_file);
  // For Runtime
  if (FLAGS_profile_mode == "runtime") {
    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return false;
    }
    double profile_time = model.GetProfileTime() * 1000;
    std::cout << "Runtime(ms): " << profile_time << "ms." << std::endl;
    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  } else {
    // For End2End
    fastdeploy::benchmark::ResourceUsageMonitor resource_moniter(
        sampling_interval, FLAGS_device_id);
    if (FLAGS_collect_memory_info) {
      resource_moniter.Start();
    }
    // Step1: warm up for warmup times
    std::cout << "Warmup " << warmup << " times..." << std::endl;
    for (int i = 0; i < warmup; i++) {
      fastdeploy::vision::DetectionResult res;
      if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return false;
      }
    }
    // Step2: repeat for repeats times
    std::cout << "Counting time..." << std::endl;
    std::cout << "Repeat " << repeats << " times..." << std::endl;
    fastdeploy::vision::DetectionResult res;
    fastdeploy::TimeCounter tc;
    tc.Start();
    for (int i = 0; i < repeats; i++) {
      if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return false;
      }
    }
    tc.End();
    double end2end = tc.Duration() / repeats * 1000;
    std::cout << "End2End(ms): " << end2end << "ms." << std::endl;
    if (FLAGS_collect_memory_info) {
      float cpu_mem = resource_moniter.GetMaxCpuMem();
      float gpu_mem = resource_moniter.GetMaxGpuMem();
      float gpu_util = resource_moniter.GetMaxGpuUtil();
      std::cout << "cpu_pss_mb: " << cpu_mem << "MB." << std::endl;
      std::cout << "gpu_pss_mb: " << gpu_mem << "MB." << std::endl;
      std::cout << "gpu_util: " << gpu_util << std::endl;
      resource_moniter.Stop();
    }
    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  }

  return true;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  int repeats = FLAGS_repeat;
  int warmup = FLAGS_warmup;
  int sampling_interval = FLAGS_sampling_interval;
  // Run model
  if (RunModel(FLAGS_model, FLAGS_image, warmup, repeats, sampling_interval) !=
      true) {
    exit(1);
  }
  return 0;
}