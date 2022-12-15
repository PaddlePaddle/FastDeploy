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
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::detection::RTMDet(model_file, params_file,
                                                     config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im_bak, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void GpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::detection::RTMDet(model_file, params_file,
                                                     config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im_bak, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void TrtInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  auto model = fastdeploy::vision::detection::RTMDet(model_file, params_file,
                                                      config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./ppyolo_dirname ./test.jpeg 0"
        << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if(std::atoi(argv[3]) == 2){
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}
