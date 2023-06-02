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

DEFINE_string(rec_label_file, "", "Path of Recognization label file of PPOCR.");
DEFINE_string(trt_shape, "1,3,48,10:4,3,48,320:8,3,48,2304",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,48,10:4,3,48,320:8,3,48,2304");

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
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shape);
    option.trt_option.SetShape("x", trt_shapes[0], trt_shapes[1],
                               trt_shapes[2]);
  }
  auto model_ppocr_rec = vision::ocr::Recognizer(
      model_file, params_file, FLAGS_rec_label_file, option, model_format);
  std::vector<std::string> model_names;
  fastdeploy::benchmark::Split(FLAGS_model, model_names, sep);
  if (model_names[model_names.size() - 1] == "ch_PP-OCRv2_rec_infer") {
    model_ppocr_rec.GetPreprocessor().SetRecImageShape({3, 32, 320});
  }
  std::string text;
  float rec_score;
  BENCHMARK_MODEL(model_ppocr_rec,
                  model_ppocr_rec.Predict(im, &text, &rec_score));
#endif
  return 0;
}