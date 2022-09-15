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

void InitAndInfer(const std::string& model_dir, const std::string& image_file,
                  const fastdeploy::RuntimeOption& option) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";

  auto model = fastdeploy::vision::detection::YOLOv7(
      model_file, params_file, option, fastdeploy::Frontend::PADDLE);
  assert(model.Initialized());

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/quant_model "
                 "path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./yolov7s_quant ./000000014439.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run on cpu with ORT "
                 "backend; 1: run "
                 "on cpu with Paddle backend ; 2: run with gpu and use "
                 "TensorRT backend."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[3]);

  if (flag == 0) {
    option.UseCpu();
    option.UseOrtBackend();
  } else if (flag == 1) {
    option.UseCpu();
    option.UsePaddleBackend();
  } else if (flag == 2) {
    option.UseGpu();
    option.UseTrtBackend();
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(model_dir, test_image, option);
  return 0;
}
