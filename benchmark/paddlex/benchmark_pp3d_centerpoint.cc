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

#include "flags.h"
#include "macros.h"
#include "option.h"

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;


static bool ReadTestPoint(const std::string &file_path,
                          std::vector<float> &data) {
  int with_timelag = 0;
  int64_t num_point_dim = 5;                      
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);
 
  if (!file_in) {
    std::cout << "Failed to read file: " << file_path << std::endl;
    return false;
  }

  std::streampos file_size;
  file_in.seekg(0, std::ios::end);
  file_size = file_in.tellg();
  file_in.seekg(0, std::ios::beg);
  
  data.resize(file_size / sizeof(float));

  file_in.read(reinterpret_cast<char *>(data.data()), file_size);
  file_in.close();

  if (file_size / sizeof(float) % num_point_dim != 0) {
    std::cout << "Loaded file size (" << file_size
            << ") is not evenly divisible by num_point_dim (" << num_point_dim
            << ")\n";
    return false;
  }
  size_t num_points = file_size / sizeof(float) / num_point_dim;
  if (!with_timelag && num_point_dim == 5 || num_point_dim > 5) {
    for (int64_t i = 0; i < num_points; ++i) {
      data[i * num_point_dim + 4] = 0.;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  std::string point_dir = FLAGS_image;
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info, false)) {
    return -1;
  }
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
    option.paddle_infer_option.collect_trt_shape_by_device = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    option.trt_option.SetShape("data", {34752, 5}, {34752, 5}, 
                                       {34752, 5});
    std::vector<float> min_input_data;
    ReadTestPoint(point_dir, min_input_data);
    // use custom data to perform collect shapes.
    option.trt_option.SetInputData("data", min_input_data);
  }
  auto model_centerpoint = vision::perception::Centerpoint(
      model_file, params_file, "", option, model_format);
  vision::PerceptionResult res;
  // Run profiling
  BENCHMARK_MODEL(model_centerpoint, model_centerpoint.Predict(point_dir, &res))
  std::cout << res.Str() << std::endl;
#endif

  return 0;
}
