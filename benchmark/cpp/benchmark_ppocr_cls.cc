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

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  // Classification Model
  auto cls_model_file = FLAGS_model + sep + "inference.pdmodel";
  auto cls_params_file = FLAGS_model + sep + "inference.pdiparams";
  if (FLAGS_backend == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (FLAGS_backend == "paddle_trt" || FLAGS_backend == "trt") {
    option.trt_option.SetShape("x", {1, 3, 48, 10}, {4, 3, 48, 320},
                               {8, 3, 48, 1024});
  }
  auto model_ppocr_cls = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, option);
  int32_t res_label;
  float res_score;
  // Run once at least
  model_ppocr_cls.Predict(im_rec, &res_label, &res_score);
  // 1. Test result diff
  std::cout << "=============== Test result diff =================\n";
  int32_t res_label_reloaded = 0;
  float res_score_reloaded = 1;
  // Calculate diff between two results.
  auto ppocr_cls_label_diff = res_label - res_label_reloaded;
  auto ppocr_cls_score_diff = res_label - res_score_reloaded;
  std::cout << "PPOCR Cls label diff: " << ppocr_cls_label_diff << std::endl;
  std::cout << "PPOCR Cls score diff: " << ppocr_cls_score_diff << std::endl;
  BENCHMARK_MODEL(model_ppocr_cls,
                  model_ppocr_cls.Predict(im_rec, &res_label, &res_score));
#endif
  return 0;
}