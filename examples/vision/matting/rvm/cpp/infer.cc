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

void CpuInfer(const std::string& model_file, const std::string& image_file,
              const std::string& background_file) {
  auto option = fastdeploy::RuntimeOption();
  auto model = fastdeploy::vision::matting::RobustVideoMatting(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();
  cv::Mat bg = cv::imread(background_file);
  fastdeploy::vision::MattingResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisMatting(im_bak, res);
  auto vis_im_with_bg =
      fastdeploy::vision::SwapBackground(im_bak, bg, res);
  cv::imwrite("visualized_result.jpg", vis_im_with_bg);
  cv::imwrite("visualized_result_fg.png", vis_im);
  std::cout << "Visualized result save in ./visualized_result.jpg "
               "and ./visualized_result_fg.jpg"
            << std::endl;
}

void GpuInfer(const std::string& model_file, const std::string& image_file,
              const std::string& background_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::matting::RobustVideoMatting(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();
  cv::Mat bg = cv::imread(background_file);
  fastdeploy::vision::MattingResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisMatting(im_bak, res);
  auto vis_im_with_bg =
      fastdeploy::vision::SwapBackground(im_bak, bg, res);
  cv::imwrite("visualized_result.jpg", vis_im_with_bg);
  cv::imwrite("visualized_result_fg.png", vis_im);
  std::cout << "Visualized result save in ./visualized_result_replaced_bg.jpg "
               "and ./visualized_result_fg.jpg"
            << std::endl;
}

void TrtInfer(const std::string& model_file, const std::string& image_file,
              const std::string& background_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.SetTrtInputShape("src", {1, 3, 1920, 1080});
  option.SetTrtInputShape("r1i", {1, 1, 1, 1}, {1, 16, 240, 135}, {1, 16, 240, 135});
  option.SetTrtInputShape("r2i", {1, 1, 1, 1}, {1, 20, 120, 68}, {1, 20, 120, 68});
  option.SetTrtInputShape("r3i", {1, 1, 1, 1}, {1, 40, 60, 34}, {1, 40, 60, 34});
  option.SetTrtInputShape("r4i", {1, 1, 1, 1}, {1, 64, 30, 17}, {1, 64, 30, 17});
  auto model = fastdeploy::vision::matting::RobustVideoMatting(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();
  cv::Mat bg = cv::imread(background_file);
  fastdeploy::vision::MattingResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisMatting(im_bak, res);
  auto vis_im_with_bg =
      fastdeploy::vision::SwapBackground(im_bak, bg, res);
  cv::imwrite("visualized_result.jpg", vis_im_with_bg);
  cv::imwrite("visualized_result_fg.png", vis_im);
  std::cout << "Visualized result save in ./visualized_result.jpg "
               "and ./visualized_result_fg.jpg"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./rvm_mobilenetv3_fp32.onnx ./test.jpg ./test_bg.jpg 0"
        << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }
  if (std::atoi(argv[4]) == 0) {
    CpuInfer(argv[1], argv[2], argv[3]);
  } else if (std::atoi(argv[4]) == 1) {
    GpuInfer(argv[1], argv[2], argv[3]);
  } else if (std::atoi(argv[4]) == 2) {
    TrtInfer(argv[1], argv[2], argv[3]);
  }
  return 0;
}
