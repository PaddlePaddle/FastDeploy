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
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto im = cv::imread(FLAGS_image);
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return -1;
  }
  auto model_file = FLAGS_model + sep + "model.pdmodel";
  auto params_file = FLAGS_model + sep + "model.pdiparams";
  auto config_file = FLAGS_model + sep + "infer_cfg.yml";
  auto model_ppyolov8 = vision::detection::PaddleYOLOv8(model_file, params_file,
                                                        config_file, option);
  vision::DetectionResult res;
  // Save result to -> disk.
  model_ppyolov8.Predict(im, &res);
  std::string det_result_path = "ppyolov8_result.txt";
  benchmark::ResultManager::SaveDetectionResult(res, det_result_path);
  // Load result from <- disk.
  vision::DetectionResult res_loaded;
  benchmark::ResultManager::LoadDetectionResult(&res_loaded, det_result_path);
  // Calculate diff between two results.
  auto det_diff =
      benchmark::ResultManager::CalculateDiffStatis(&res, &res_loaded);
  std::cout << "diff: mean=" << det_diff.mean << ",max=" << det_diff.max
            << ",min=" << det_diff.min << std::endl;
  BENCHMARK_MODEL(model_ppyolov8, model_ppyolov8.Predict(im, &res))
  auto vis_im = vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  return 0;
}