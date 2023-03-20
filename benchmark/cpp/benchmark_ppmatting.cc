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

DEFINE_string(trt_shape, "1,3,512,512:1,3,512,512:1,3,512,512",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,512,512:1,3,512,512:1,3,512,512");

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
                               &model_format, config_info)) {
    return -1;
  }

  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;
  auto config_file = FLAGS_model + sep + config_name;
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shape);
    option.trt_option.SetShape("img", trt_shapes[0], trt_shapes[1],
                               trt_shapes[2]);
  }
  auto model_ppmatting = vision::matting::PPMatting(
      model_file, params_file, config_file, option, model_format);
  vision::MattingResult res;
  if (config_info["precision_compare"] == "true") {
    // Run once at least
    model_ppmatting.Predict(&im, &res);
    // 1. Test result diff
    std::cout << "=============== Test result diff =================\n";
    // Save result to -> disk.
    std::string matting_result_path = "ppmatting_result.txt";
    benchmark::ResultManager::SaveMattingResult(res, matting_result_path);
    // Load result from <- disk.
    vision::MattingResult res_loaded;
    benchmark::ResultManager::LoadMattingResult(&res_loaded,
                                                matting_result_path);
    // Calculate diff between two results.
    auto matting_diff =
        benchmark::ResultManager::CalculateDiffStatis(res, res_loaded);
    std::cout << "Alpha diff: mean=" << matting_diff.alpha.mean
              << ", max=" << matting_diff.alpha.max
              << ", min=" << matting_diff.alpha.min << std::endl;
    if (res_loaded.contain_foreground) {
      std::cout << "Foreground diff: mean=" << matting_diff.foreground.mean
                << ", max=" << matting_diff.foreground.max
                << ", min=" << matting_diff.foreground.min << std::endl;
    }
  }
  BENCHMARK_MODEL(model_ppmatting, model_ppmatting.Predict(&im, &res))
  auto vis_im = vision::VisMatting(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif
  return 0;
}
