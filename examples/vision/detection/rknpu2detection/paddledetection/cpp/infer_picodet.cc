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

void InferPicodet(const std::string& device = "cpu");

int main() {
  InferPicodet("npu");
  return 0;
}

fastdeploy::RuntimeOption GetOption(const std::string& device) {
  auto option = fastdeploy::RuntimeOption();
  if (device == "npu") {
    option.UseRKNPU2();
  } else {
    option.UseCpu();
  }
  return option;
}

fastdeploy::ModelFormat GetFormat(const std::string& device) {
  auto format = fastdeploy::ModelFormat::ONNX;
  if (device == "npu") {
    format = fastdeploy::ModelFormat::RKNN;
  } else {
    format = fastdeploy::ModelFormat::ONNX;
  }
  return format;
}

std::string GetModelPath(std::string& model_path, const std::string& device) {
  if (device == "npu") {
    model_path += "rknn";
  } else {
    model_path += "onnx";
  }
  return model_path;
}

void InferPicodet(const std::string &device) {
  std::string model_file = "./model/picodet_s_416_coco_npu/picodet_s_416_coco_npu_rk3588.";
  std::string params_file;
  std::string config_file = "./model/picodet_s_416_coco_npu/infer_cfg.yml";

  fastdeploy::RuntimeOption option = GetOption(device);
  fastdeploy::ModelFormat format = GetFormat(device);
  model_file = GetModelPath(model_file, device);
  auto model = fastdeploy::vision::detection::RKPicoDet(
      model_file, params_file, config_file,option,format);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  auto image_file = "./images/000000014439.jpg";
  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  clock_t start = clock();
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  clock_t end = clock();
  auto dur = static_cast<double>(end - start);
  printf("picodet_npu use time:%f\n", (dur / CLOCKS_PER_SEC));

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res,0.5);
  cv::imwrite("picodet_npu_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./picodet_npu_result.jpg" << std::endl;
}