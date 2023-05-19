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

void InitAndInfer(const std::string& model_dir, const std::string& images_dir,
                  const fastdeploy::RuntimeOption& option) {
  auto model_file = model_dir + sep + "petrv2_inference.pdmodel";
  auto params_file = model_dir + sep + "petrv2_inference.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";
  fastdeploy::vision::EnableFlyCV();
  auto model = fastdeploy::vision::perception::Petr(
      model_file, params_file, config_file, option,
      fastdeploy::ModelFormat::PADDLE);
  assert(model.Initialized());

  std::vector<cv::Mat> im_batch;
  for (int i = 0; i < 12; i++) {
    auto image_file = images_dir + sep + "image" + std::to_string(i) + ".png";
    auto im = cv::imread(image_file);
    im_batch.emplace_back(im);
  }

  std::vector<fastdeploy::vision::PerceptionResult> res;
  if (!model.BatchPredict(im_batch, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  std::cout << res[0].Str() << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/paddle_model"
                 "path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./petr ./00000.png 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with paddle-trt"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  if (std::atoi(argv[3]) == 0) {
    option.UseCpu();
  } else if (std::atoi(argv[3]) == 1) {
    option.UseGpu();
  } else if (std::atoi(argv[3]) == 2) {
    option.UseGpu();
    option.UseTrtBackend();
    option.EnablePaddleToTrt();
    option.SetTrtInputShape("images", {1, 3, 384, 1280});
    option.SetTrtInputShape("down_ratios", {1, 2});
    option.SetTrtInputShape("trans_cam_to_img", {1, 3, 3});
    option.SetTrtInputData("trans_cam_to_img",
                           {721.53771973, 0., 609.55932617, 0., 721.53771973,
                            172.85400391, 0, 0, 1});
    option.EnablePaddleTrtCollectShape();
  }
  option.UsePaddleBackend();

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(model_dir, test_image, option);
  return 0;
}
