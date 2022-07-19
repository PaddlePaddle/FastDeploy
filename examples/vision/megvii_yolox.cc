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

int main() {
  namespace vis = fastdeploy::vision;

  std::string model_file = "../resources/models/yolox_s.onnx";
  std::string img_path = "../resources/images/bus.jpg";
  std::string vis_path  = "../resources/outputs/megvii_yolox_vis_result.jpg";

  auto model = vis::megvii::YOLOX(model_file);
  if (!model.Initialized()) {
    std::cerr << "Init Failed! Model: " << model_file << std::endl;
    return -1;
  } else {
    std::cout << "Init Done! Model:" << model_file << std::endl; 
  }
  model.EnableDebug();
  
  cv::Mat im = cv::imread(img_path);
  cv::Mat vis_im = im.clone();

  vis::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Prediction Failed." << std::endl;
    return -1;
  } else {
    std::cout << "Prediction Done!" << std::endl; 
  }

  // 输出预测框结果
  std::cout << res.Str() << std::endl;

  // 可视化预测结果
  vis::Visualize::VisDetection(&vis_im, res);
  cv::imwrite(vis_path, vis_im);
  std::cout << "Detect Done! Saved: " << vis_path << std::endl;
  return 0;
}
