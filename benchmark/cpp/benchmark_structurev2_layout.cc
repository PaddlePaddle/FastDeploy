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
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    option.trt_option.SetShape("image", {1, 3, 800, 608}, {1, 3, 800, 608},
                               {1, 3, 800, 608});
  }
  auto layout_model = vision::ocr::StructureV2Layout(model_file, params_file,
                                                     option, model_format);
  // 5 for publaynet, 10 for cdla
  layout_model.GetPostprocessor().SetNumClass(5);

  vision::DetectionResult res;
  if (config_info["precision_compare"] == "true") {
    // Run once at least
    layout_model.Predict(im, &res);
    // 1. Test result diff
    std::cout << "=============== Test result diff =================\n";
    // Save result to -> disk.
    std::string layout_result_path = "layout_result.txt";
    benchmark::ResultManager::SaveDetectionResult(res, layout_result_path);
    // Load result from <- disk.
    vision::DetectionResult res_loaded;
    benchmark::ResultManager::LoadDetectionResult(&res_loaded,
                                                  layout_result_path);
    // Calculate diff between two results.
    auto det_diff =
        benchmark::ResultManager::CalculateDiffStatis(res, res_loaded);
    std::cout << "Boxes diff: mean=" << det_diff.boxes.mean
              << ", max=" << det_diff.boxes.max
              << ", min=" << det_diff.boxes.min << std::endl;
    std::cout << "Label_ids diff: mean=" << det_diff.labels.mean
              << ", max=" << det_diff.labels.max
              << ", min=" << det_diff.labels.min << std::endl;
  }
  // Run profiling
  BENCHMARK_MODEL(layout_model, layout_model.Predict(im, &res))
  std::vector<std::string> labels = {"text", "title", "list", "table",
                                     "figure"};
  if (layout_model.GetPostprocessor().GetNumClass() == 10) {
    labels = {"text",      "title",         "figure", "figure_caption",
              "table",     "table_caption", "header", "footer",
              "reference", "equation"};
  }
  auto vis_im =
      vision::VisDetection(im, res, labels, 0.3, 2, .5f, {255, 0, 0}, 2);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif
  return 0;
}