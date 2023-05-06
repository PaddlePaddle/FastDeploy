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

void HorizonInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + "/ppyoloe_640x640_rgb_s.bin";
  auto params_file = "";
  auto config_file = model_dir + "/infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseHorizon();
  option.UseHorizonNPUBackend();

  auto format = fastdeploy::ModelFormat::HORIZON;

  auto model = fastdeploy::vision::detection::PPYOLOE(
      model_file, params_file, config_file, option, format);

  model.GetPreprocessor().DisablePermute();
  model.GetPreprocessor().DisableNormalize();
  model.GetPostprocessor().ApplyNMS();
  fastdeploy::vision::DetectionResult res;
  auto im = cv::imread(image_file);

  fastdeploy::TimeCounter tc;
  tc.Start();
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  tc.End();
  tc.PrintInfo("PPDet in Horizon");

  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("infer_horizon.jpg", vis_im);
  std::cout << "Visualized result saved in ./infer_horizon.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: infer_ppyoloe_demo path/to/model_dir path/to/image, "
                 "e.g ./infer_ppyoloe_demo ./ppyoloe_model_dir ./test.jpeg"
              << std::endl;
    return -1;
  }

  HorizonInfer(argv[1], argv[2]);

  return 0;
}
