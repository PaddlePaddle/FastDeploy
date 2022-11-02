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
DEFINE_string(backend, "default",
              "The inference runtime backend, support: ['default', 'ort', "
              "'paddle', 'ov', 'trt', 'paddle_trt']");
DEFINE_bool(use_fp16, false, "Whether to use FP16 mode, only support 'trt' and 'paddle_trt' backend");

void PrintUsage() {
  std::cout << "Usage: infer_demo --model model_path --image img_path --device [cpu|gpu] --backend "
               "[default|ort|paddle|ov|trt|paddle_trt] "
               "--use_fp16 false"
            << std::endl;
  std::cout << "Default value of device: cpu" << std::endl;
  std::cout << "Default value of backend: default" << std::endl;
  std::cout << "Default value of use_fp16: false" << std::endl;
}

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  if (FLAG_device == "gpu") {
    option->UseGpu();
    if (FLAG_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleBackend();
    } else if (FLAGS_backend == "trt" || 
               FLAGS_backend == "paddle_trt") {
      option->UseTrtBackend();
      option->SetTrtInputShape("input", {1, 3, 112, 112});
      if (FLAGS_backend == "paddle_trt") {
        option->EnablePaddleToTrt();
      }
      if (FLAGS_use_fp16) {
        option->EnableTrtFP16();
      }
    } else if (FLAGS_backend == "default") {
      return true;
    } else {
      std::cout << "While inference with GPU, only support default/ort/paddle/trt/paddle_trt now, " << FLAG_backend << " is not supported." << std::endl;
      return false;
    }
  } else if (FLAG_device == "cpu") {
    if (FLAGS_backend == "ort") {
      option->UseOrtBackend();
    } else if (FLAGS_backend == "ov") {
      option->UseOpenVINOBackend();
    } else if (FLAGS_backend == "paddle") {
      option->UsePaddleBackend();
    } else if (FLAGS_backend = "default") {
      return true;
    } else {
      std::cout << "While inference with CPU, only support default/ort/ov/paddle now, " << FLAG_backend << " is not supported." << std::endl;
      return false;
    }
  } else {
    std::cerr << "Only support device CPU/GPU now, "  << FLAG_device << " is not supported." << std::endl;
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

  auto model = fastdeploy::vision::facealign::PFLD(FLAGS_model, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return -1;
  }

  auto im = cv::imread(FLAGS_image);
  auto im_bak = im.clone();

  fastdeploy::vision::FaceAlignmentResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return -1;
  }
  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisFaceAlignment(im_bak, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  
  return 0;
}
