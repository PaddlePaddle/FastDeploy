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
#include <iostream>
#include <string>
#include "fastdeploy/vision.h"
#include <sys/time.h>
void RKNPU2Infer(const std::string& model_dir, const std::string& image_file);

void RKNPU2Infer(const std::string& model_dir, const std::string& image_file) {
  struct timeval start_time, stop_time;
  auto model_file = model_dir + "/picodet_s_416_coco_lcnet_rk3588.rknn";
  auto params_file = "";
  auto config_file = model_dir + "/infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();

  auto format = fastdeploy::ModelFormat::RKNN;

  auto model = fastdeploy::vision::detection::PicoDet(
      model_file, params_file, config_file,option,format);

  model.GetPostprocessor().ApplyDecodeAndNMS();

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  fastdeploy::TimeCounter tc;
  tc.Start();
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  tc.End();
  std::cout << tc.Duration() << std::endl;

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res,0.5);
  cv::imwrite("picodet_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./picodet_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./picodet_model_dir ./test.jpeg"
        << std::endl;
    return -1;
  }

  RKNPU2Infer(argv[1], argv[2]);

  return 0;
}

