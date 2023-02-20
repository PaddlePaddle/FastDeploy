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
  auto model_ppseg = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);
  fastdeploy::vision::SegmentationResult res;
  BENCHMARK_MODEL(model_ppseg, model_ppseg.Predict(im, &res))
  return 0;
}