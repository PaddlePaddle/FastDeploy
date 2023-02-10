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

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

bool RunModel(std::string model_dir, std::string image_file, size_t warmup,
              size_t repeats, size_t dump_period, std::string cpu_mem_file_name,
              std::string gpu_mem_file_name) {
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return false;
  }
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  if (FLAGS_profile_mode == "runtime") {
    option.EnableProfiling(FLAGS_include_h2d_d2h, repeats, warmup);
  }
  auto model = fastdeploy::vision::detection::PaddleYOLOv8(
      model_file, params_file, config_file, option);
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
    // Step1: warm up for warmup times
    std::cout << "Warmup " << warmup << " times..." << std::endl;
    for (int i = 0; i < warmup; i++) {
      fastdeploy::vision::DetectionResult res;
      if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return false;
      }
    }
    std::vector<float> end2end_statis;
    // Step2: repeat for repeats times
    std::cout << "Counting time..." << std::endl;
    fastdeploy::TimeCounter tc;
    fastdeploy::vision::DetectionResult res;
    for (int i = 0; i < repeats; i++) {
      if (FLAGS_collect_memory_info && i % dump_period == 0) {
        fastdeploy::benchmark::DumpCurrentCpuMemoryUsage(cpu_mem_file_name);
#if defined(WITH_GPU)
        fastdeploy::benchmark::DumpCurrentGpuMemoryUsage(gpu_mem_file_name,
                                                         FLAGS_device_id);
#endif
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
    std::cout << "End2End(ms): " << end2end << "ms." << std::endl;
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
  int dump_period = FLAGS_dump_period;
  std::string cpu_mem_file_name = "result_cpu.txt";
  std::string gpu_mem_file_name = "result_gpu.txt";
  // Run model
  if (RunModel(FLAGS_model, FLAGS_image, warmup, repeats, dump_period,
               cpu_mem_file_name, gpu_mem_file_name) != true) {
    exit(1);
  }
  if (FLAGS_collect_memory_info) {
    float cpu_mem = fastdeploy::benchmark::GetCpuMemoryUsage(cpu_mem_file_name);
    std::cout << "cpu_pss_mb: " << cpu_mem << "MB." << std::endl;
#if defined(WITH_GPU)
    float gpu_mem = fastdeploy::benchmark::GetGpuMemoryUsage(gpu_mem_file_name);
    std::cout << "gpu_pss_mb: " << gpu_mem << "MB." << std::endl;
#endif
  }
  return 0;
}
