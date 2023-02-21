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
  auto model_file = FLAGS_model + sep + "model.pdmodel";
  auto params_file = FLAGS_model + sep + "model.pdiparams";
  auto config_file = FLAGS_model + sep + "deploy.yaml";
  if (FLAGS_backend == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (FLAGS_backend == "paddle_trt" || FLAGS_backend == "trt") {
    option.trt_option.SetShape("x", {1, 3, 192, 192}, {1, 3, 192, 192},
                               {1, 3, 192, 192});
  }
  auto model_ppseg = vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);
  vision::SegmentationResult res;
  // Run once at least
  model_ppseg.Predict(im, &res);
  // 1. Test result diff
  std::cout << "=============== Test result diff =================\n";
  // Save result to -> disk.
  std::string seg_result_path = "ppseg_result.txt";
  benchmark::ResultManager::SaveSegmentationResult(res, seg_result_path);
  // Load result from <- disk.
  vision::SegmentationResult res_loaded;
  benchmark::ResultManager::LoadSegmentationResult(&res_loaded,
                                                   seg_result_path);
  // Calculate diff between two results.
  auto seg_diff =
      benchmark::ResultManager::CalculateDiffStatis(res, res_loaded);
  std::cout << "Labels diff: mean=" << seg_diff.labels.mean
            << ", max=" << seg_diff.labels.max
            << ", min=" << seg_diff.labels.min << std::endl;
  std::cout << "Scores diff: mean=" << seg_diff.scores.mean
            << ", max=" << seg_diff.scores.max
            << ", min=" << seg_diff.scores.min << std::endl;
  BENCHMARK_MODEL(model_ppseg, model_ppseg.Predict(im, &res))
  auto vis_im = vision::VisSegmentation(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif
  return 0;
}