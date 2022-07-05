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
  auto model = fastdeploy::vision::ultralytics::YOLOv5("yolov5s.onnx");
  model.EnableDebug();
  if (!model.Initialized()) {
    std::cout << "Init Failed." << std::endl;
    return -1;
  }
  cv::Mat im = cv::imread("bus.jpg");

  for (size_t i = 0; i < 10; ++i) {
    auto im1 = im.clone();
    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(&im1, &res)) {
      std::cout << "Predict Failed." << std::endl;
      return -1;
    }
  }

  {
    fastdeploy::vision::DetectionResult res;
    auto vis_im = im.clone();
    if (!model.Predict(&im, &res)) {
      std::cout << "Predict Failed." << std::endl;
      return -1;
    }

    fastdeploy::vision::Visualize::VisDetection(&vis_im, res);
    cv::imwrite("vis.jpg", vis_im);
    // Print Detection Result
    std::cout << res.Str() << std::endl;
  }
    return 0;
}
