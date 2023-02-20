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
  // Set max_batch_size 1 for best performance
  if (FLAGS_backend == "paddle_trt") {
    option.trt_option.max_batch_size = 1;
  }
  auto model_file = FLAGS_model + sep + "inference.pdmodel";
  auto params_file = FLAGS_model + sep + "inference.pdiparams";
  auto config_file = FLAGS_model + sep + "inference_cls.yaml";
  auto model_ppcls = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  fastdeploy::vision::ClassifyResult res;
  // Run once at least
  model_ppcls.Predict(im, &res);
  // 1. Test result diff
  std::cout << "=============== Test result diff =================\n";
  // Save result to -> disk.
  std::string cls_result_path = "ppcls_result.txt";
  benchmark::ResultManager::SaveClassifyResult(res, cls_result_path);
  // Load result from <- disk.
  vision::ClassifyResult res_loaded;
  benchmark::ResultManager::LoadClassifyResult(&res_loaded, cls_result_path);
  // Calculate diff between two results.
  auto cls_diff =
      benchmark::ResultManager::CalculateDiffStatis(&res, &res_loaded);
  std::cout << "Labels diff: mean=" << cls_diff.labels.mean
            << ", max=" << cls_diff.labels.max
            << ", min=" << cls_diff.labels.min << std::endl;
  std::cout << "Scores diff: mean=" << cls_diff.scores.mean
            << ", max=" << cls_diff.scores.max
            << ", min=" << cls_diff.scores.min << std::endl;
  BENCHMARK_MODEL(model_ppcls, model_ppcls.Predict(im, &res))
#endif
  return 0;
}