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

void HorizonInfer(const std::string &model_dir, const std::string &image_file) {
  auto model_file = model_dir + "/resnet_50_224x224_rgb.bin";
  auto params_file = "";
  auto config_file = model_dir + "/inference_cls.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseHorizon();
  option.UseHorizonNPUBackend();

  auto format = fastdeploy::ModelFormat::HORIZON;

  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option, format);
  if (!model.Initialized()) {
    std::cerr << "model Failed to initialize." << std::endl;
    return;
  }
  model.GetPreprocessor().DisablePermute();
  model.GetPreprocessor().DisableNormalize();

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  fastdeploy::TimeCounter tc;
  tc.Start();
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  tc.End();
  tc.PrintInfo("PPClas in Horizon");
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: HorizonInfer path/to/model_dir path/to/image, "
                 "e.g ./infer_demo ./ppclas_model_dir "
                 "./images/ILSVRC2012_val_00000010.jpeg"
              << std::endl;
    return -1;
  }
  HorizonInfer(argv[1], argv[2]);
  return 0;
}
