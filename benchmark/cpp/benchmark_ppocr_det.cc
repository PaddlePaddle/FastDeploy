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
  // Detection Model
  auto det_model_file = FLAGS_model + sep + "inference.pdmodel";
  auto det_params_file = FLAGS_model + sep + "inference.pdiparams";
  if (FLAGS_backend == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (FLAGS_backend == "paddle_trt" || FLAGS_backend == "trt") {
    option.trt_option.SetShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                               {1, 3, 960, 960});
  }
  auto model_ppocr_det = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, option);
  std::vector<std::array<int, 8>> res;
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
  BENCHMARK_MODEL(model_ppocr_det, model_ppocr_det.Predict(im, &boxes_result));
#endif
  return 0;
}