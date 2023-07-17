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

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
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
  std::vector<float> cam_data{7.183351e+02, 0.000000e+00,  6.003891e+02,
                              4.450382e+01, 0.000000e+00,  7.183351e+02,
                              1.815122e+02, -5.951107e-01, 0.000000e+00,
                              0.000000e+00, 1.000000e+00,  2.616315e-03};
  std::vector<float> lidar_data = {
      0.0048523,   -0.9999298, -0.01081266, -0.00711321,
      -0.00302069, 0.01079808, -0.99993706, -0.06176636,
      0.99998367,  0.00488465, -0.00296808, -0.26739058,
      0.,          0.,         0.,          1.};
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
    option.paddle_infer_option.collect_trt_shape_by_device = true;
    option.paddle_infer_option.trt_min_subgraph_size = 12;
    option.paddle_infer_option.DisableTrtOps({"squeeze2"});
    option.trt_option.max_batch_size = 1;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    // use custom data to perform collect shapes.
    option.trt_option.SetShape("images", {1, 3, 375, 1242}, {1, 3, 375, 1242},
                               {1, 3, 375, 1242});
    option.trt_option.SetShape("trans_lidar_to_cam", {1, 4, 4}, {1, 4, 4},
                               {1, 4, 4});
    option.trt_option.SetShape("trans_cam_to_img", {1, 3, 4}, {1, 3, 4},
                               {1, 3, 4});
    std::vector<float> image_data;
    image_data.assign(im.data, im.data + 1 * 3 * 375 * 1242);
    option.trt_option.SetInputData("trans_lidar_to_cam", lidar_data);
    option.trt_option.SetInputData("trans_cam_to_img", cam_data);
    option.trt_option.SetInputData("images", image_data);
  }
  auto model_cadnn = vision::perception::Caddn(model_file, params_file, "",
                                               option, model_format);
  vision::PerceptionResult res;
  // Run profiling
  BENCHMARK_MODEL(model_cadnn,
                  model_cadnn.Predict(im, cam_data, lidar_data, &res))
  std::cout << res.Str() << std::endl;
#endif

  return 0;
}
