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

#include <gflags/gflags.h>
#include "fastdeploy/vision.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_file, "", "Path of input image");
DEFINE_string(device, "CPU",
              "Choose the device you want to run, it can be: CPU/GPU, "
              "default is CPU.");
DEFINE_string(backend, "default",
              "Set inference backend, support one of ['default', 'ort', "
              "'paddle', 'trt', 'openvino']");

void CpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();

  if (FLAGS_backend == "ort") {
    option.UseOrtBackend();
  } else if (FLAGS_backend == "paddle") {
    option.UsePaddleBackend();
  } else if (FLAGS_backend == "trt") {
    std::cerr << "Use --backend=trt for inference must set --device=gpu"
              << std::endl;
    return;
  } else if (FLAGS_backend == "openvino") {
    option.UseOpenVINOBackend();
  } else if (FLAGS_backend == "default") {
    std::cout << "Use default backend for inference" << std::endl;
  } else {
    std::cerr << "Don't support backend type: " + FLAGS_backend << std::endl;
    return;
  }
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file,
                                                      config_file);
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
  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void GpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  if (FLAGS_backend == "ort") {
    option.UseOrtBackend();
  } else if (FLAGS_backend == "paddle") {
    option.UsePaddleBackend();
  } else if (FLAGS_backend == "trt") {
    option.UseTrtBackend();
    option.SetTrtInputShape("image", {1, 3, 640, 640});
    option.SetTrtInputShape("scale_factor", {1, 2});
  } else if (FLAGS_backend == "openvino") {
    std::cerr << "Use --backend=openvino for inference must set --device=cpu"
              << std::endl;
    return
  } else if (FLAGS_backend == "default") {
    std::cout << "Use default backend for inference" << std::endl;
  } else {
    std::cerr << "Don't support backend type: " + FLAGS_backend << std::endl;
    return;
  }

  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file,
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
  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty() || FLAGS_image_file.empty()) {
    std::cout << "Usage: infer_demo --model_dir=/path/to/model_dir "
                 "--image_file=/path/to/image --device=device, "
                 "e.g ./infer_model --model_dir=./ppyoloe_model_dir "
                 "--image_file=./test.jpeg --device=cpu"
              << std::endl;
    std::cout << "For more information, use ./infer_model --help" << std::endl;
    return -1;
  }

  if (FLAGS_device == "cpu") {
    CpuInfer(FLAGS_model_dir, FLAGS_image_file);
  } else if (FLAGS_device == "gpu") {
    GpuInfer(FLAGS_model_dir, FLAGS_image_file);
  } else {
    std::cerr << "Don't support device type:" + FLAGS_device << std::endl;
    return -1;
  }
  return 0;
}
