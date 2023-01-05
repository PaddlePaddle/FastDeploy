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
#include "gflags/gflags.h"

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(image, "", "Path of the image file.");
DEFINE_string(device, "cpu",
              "Type of inference device, support 'cpu' or 'gpu'.");

void PrintUsage() {
  std::cout << "Usage: infer_demo --model model_path --image img_path --device [cpu|gpu]"
            << std::endl;
  std::cout << "Default value of device: cpu" << std::endl;
}

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  if (FLAGS_device == "gpu") {
    option->UseGpu();
    } 
  else if (FLAGS_device == "cpu") {
    option->SetPaddleMKLDNN(false);
    return true;
  } else {
    std::cerr << "Only support device CPU/GPU now, "  << FLAGS_device << " is not supported." << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return -1;
  }

  auto model = fastdeploy::vision::generation::AnimeGAN(FLAGS_model+"/model.pdmodel", FLAGS_model+"/model.pdiparams", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return -1;
  }

  auto im = cv::imread(FLAGS_image);
  cv::Mat res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return -1;
  }

  cv::imwrite("style_transfer_result.png", res);
  std::cout << "Visualized result saved in ./style_transfer_result.png" << std::endl;
  
  return 0;
}
