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
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // print res
  std::cout << res.Str() << std::endl;
}

void GpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // print res
  std::cout << res.Str() << std::endl;
}

void TrtInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.SetTrtInputShape("inputs", {1, 3, 224, 224}, {1, 3, 224, 224},
                          {1, 3, 224, 224});
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // print res
  std::cout << res.Str() << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_demo ./ResNet50_vd ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}
