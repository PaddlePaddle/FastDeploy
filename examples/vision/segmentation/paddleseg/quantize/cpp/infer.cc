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
  auto config_file = model_dir + sep + "deploy.yaml";

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file,option);

  assert(model.Initialized());

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;

}

// int main(int argc, char* argv[]) {
//   if (argc < 3) {
//     std::cout
//         << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
//            "e.g ./infer_model ./ppseg_model_dir ./test.jpeg 0"
//         << std::endl;
//     std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
//                  "with gpu; 2: run with gpu and use tensorrt backend."
//               << std::endl;
//     return -1;
//   }

//   fastdeploy::RuntimeOption option;
//   option.UseCpu();
//   option.UsePaddleBackend();
//   std::cout<<"Xyy-debug, enable Paddle Backend==!";

//   std::string model_dir = argv[1];
//   std::string test_image = argv[2];
//   InitAndInfer(model_dir, test_image, option);
//   return 0;
// }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/quant_model "
                 "path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ResNet50_vd_quant ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run on cpu with ORT "
                 "backend; 1: run "
                 "on gpu with TensorRT backend. "
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[3]);

  if (flag == 0) {
    option.UseCpu();
    option.UseOrtBackend();
    std::cout<<"Use ORT!"<<std::endl;
  } else if (flag == 1) {
    option.UseCpu();
    option.UsePaddleBackend();
    std::cout<<"Use PP!"<<std::endl;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(model_dir, test_image, option);
  return 0;
}