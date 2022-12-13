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

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

DEFINE_string(model, "", "Directory of the inference model");
DEFINE_string(image, "", "Path of the image file.");

DEFINE_string(device, "cpu", "Type of openvino device, 'cpu' or 'intel_gpu'");

void InitAndInfer(const std::string& model_dir, const std::string& image_file, const fastdeploy::RuntimeOption& option) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto model = fastdeploy::vision::detection::PPYOLOE(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  std::cout << "Warmup 20 times..." << std::endl;
  for (int i = 0; i < 20; ++i) {
    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
  }

  std::cout << "Counting time..." << std::endl;
  fastdeploy::TimeCounter tc;
  tc.Start();
  for (int i = 0; i < 50; ++i) {
    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
  }
  tc.End();
  std::cout << "Elapsed time: " << tc.Duration() * 1000 << "ms." << std::endl;

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  cv::Mat vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

fastdeploy::RuntimeOption BuildOption(const std::string& device) {
  if (device != "cpu" && device != "intel_gpu") {
    std::cerr << "The flag device only can be 'cpu' or 'intel_gpu'" << std::endl;
    std::abort();
  }
  fastdeploy::RuntimeOption option;
  option.UseOpenVINOBackend();
  if (device == "intel_gpu") {
    option.SetOpenVINODevice("HETERO:GPU,CPU");
    std::map<std::string, std::vector<int64_t>> shape_info;
    shape_info["image"] = {1, 3, 640, 640};
    shape_info["scale_factor"] = {1, 2};
    option.SetOpenVINOShapeInfo(shape_info);
    option.SetOpenVINOCpuOperators({"MulticlassNms"});
  }
  return option;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto option = BuildOption(FLAGS_device);
  InitAndInfer(FLAGS_model, FLAGS_image, option);
  return 0;
}
