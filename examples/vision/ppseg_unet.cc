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
#include "yaml-cpp/yaml.h"

int main() {
  namespace vis = fastdeploy::vision;

  std::string model_file = "../resources/models/unet_Cityscapes/model.pdmodel";
  std::string params_file =
      "../resources/models/unet_Cityscapes/model.pdiparams";
  std::string config_file = "../resources/models/unet_Cityscapes/deploy.yaml";
  std::string img_path = "../resources/images/cityscapes_demo.png";
  std::string vis_path = "../resources/outputs/vis.jpeg";

  auto model = vis::ppseg::Model(model_file, params_file, config_file);
  if (!model.Initialized()) {
    std::cerr << "Init Failed." << std::endl;
    return -1;
  }

  cv::Mat im = cv::imread(img_path);
  cv::Mat vis_im;

  vis::SegmentationResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Prediction Failed." << std::endl;
    return -1;
  } else {
    std::cout << "Prediction Done!" << std::endl;
  }

  // 输出预测框结果
  std::cout << res.Str() << std::endl;

  YAML::Node cfg = YAML::LoadFile(config_file);
  int num_classes = 19;
  if (cfg["Deploy"]["num_classes"]) {
    num_classes = cfg["Deploy"]["num_classes"].as<int>();
  }

  // 可视化预测结果
  vis::Visualize::VisSegmentation(im, res, &vis_im, num_classes);
  cv::imwrite(vis_path, vis_im);
  std::cout << "Inference Done! Saved: " << vis_path << std::endl;
  return 0;
}
