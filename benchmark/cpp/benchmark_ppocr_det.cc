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

DEFINE_string(trt_shape, "1,3,64,64:1,3,640,640:1,3,960,960",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,64,64:1,3,640,640:1,3,960,960");

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
  // Detection Model
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info, false)) {
    return -1;
  }
  // Classification Model
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shape);
    option.trt_option.SetShape("x", trt_shapes[0], trt_shapes[1],
                               trt_shapes[2]);
  }
  auto model_ppocr_det =
      vision::ocr::DBDetector(model_file, params_file, option, model_format);
  std::vector<std::array<int, 8>> res;
  if (config_info["precision_compare"] == "true") {
    // Run once at least
    model_ppocr_det.Predict(im, &res);
    // 1. Test result diff
    std::cout << "=============== Test result diff =================\n";
    // Save result to -> disk.
    std::string ppocr_det_result_path = "ppocr_det_result.txt";
    benchmark::ResultManager::SaveOCRDetResult(res, ppocr_det_result_path);
    // Load result from <- disk.
    std::vector<std::array<int, 8>> res_loaded;
    benchmark::ResultManager::LoadOCRDetResult(&res_loaded,
                                               ppocr_det_result_path);
    // Calculate diff between two results.
    auto ppocr_det_diff =
        benchmark::ResultManager::CalculateDiffStatis(res, res_loaded);
    std::cout << "PPOCR Boxes diff: mean=" << ppocr_det_diff.boxes.mean
              << ", max=" << ppocr_det_diff.boxes.max
              << ", min=" << ppocr_det_diff.boxes.min << std::endl;
  }
  BENCHMARK_MODEL(model_ppocr_det, model_ppocr_det.Predict(im, &res));
#endif
  return 0;
}