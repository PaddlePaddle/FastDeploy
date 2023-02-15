// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision.h"

void RKNPU2Infer(const std::string& tinypose_model_dir,
                 const std::string& image_file) {
  auto tinypose_model_file =
      tinypose_model_dir + "/picodet_s_416_coco_lcnet_rk3588.rknn";
  auto tinypose_params_file = "";
  auto tinypose_config_file = tinypose_model_dir + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();
  auto tinypose_model = fastdeploy::vision::keypointdetection::PPTinyPose(
      tinypose_model_file, tinypose_params_file, tinypose_config_file, option);

  if (!tinypose_model.Initialized()) {
    std::cerr << "TinyPose Model Failed to initialize." << std::endl;
    return;
  }

  tinypose_model.DisablePermute();
  tinypose_model.DisableNormalize();

  auto im = cv::imread(image_file);
  fastdeploy::vision::KeyPointDetectionResult res;
  if (!tinypose_model.Predict(&im, &res)) {
    std::cerr << "TinyPose Prediction Failed." << std::endl;
    return;
  } else {
    std::cout << "TinyPose Prediction Done!" << std::endl;
  }

  std::cout << res.Str() << std::endl;

  auto tinypose_vis_im = fastdeploy::vision::VisKeypointDetection(im, res, 0.5);
  cv::imwrite("tinypose_vis_result.jpg", tinypose_vis_im);
  std::cout << "TinyPose visualized result saved in ./tinypose_vis_result.jpg"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/pptinypose_model_dir path/to/image "
                 "run_option, "
                 "e.g ./infer_model ./pptinypose_model_dir ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend; 3: run "
                 "with kunlunxin."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    RKNPU2Infer(argv[1], argv[2]);
  }
  return 0;
}
