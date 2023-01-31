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
DEFINE_int64(topk, 1, "Topk classify result of the image file");

DEFINE_string(device, "cpu", "Type of openvino device, 'cpu' or 'intel_gpu'");

void InitAndInfer(const std::string& model_dir, const std::string& image_file, int topk, const fastdeploy::RuntimeOption& option) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";

  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);

  model.GetPostprocessor().SetTopk(topk);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  std::cout << "Warmup 20 times..." << std::endl;
  for (int i = 0; i < 20; ++i) {
    fastdeploy::vision::ClassifyResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
  }

  std::cout << "Counting time..." << std::endl;
  fastdeploy::TimeCounter tc;
  tc.Start();
  for (int i = 0; i < 50; ++i) {
    fastdeploy::vision::ClassifyResult res;
    if (!model.Predict(im, &res)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
  }
  tc.End();
  std::cout << "Elapsed time: " << tc.Duration() * 1000 << "ms." << std::endl;


  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  // print res
  std::cout << res.Str() << std::endl;
}

fastdeploy::RuntimeOption BuildOption(const std::string& device) {
  if (device != "cpu" && device != "intel_gpu") {
    std::cerr << "The flag device only can be 'cpu' or 'intel_gpu'" << std::endl;
    std::abort();
  }
  fastdeploy::RuntimeOption option;
  option.UseOpenVINOBackend();
  if (device == "intel_gpu") {
    option.SetOpenVINODevice("GPU");
    std::map<std::string, std::vector<int64_t>> shape_info;
    shape_info["inputs"] = {1, 3, 224, 224};
    option.SetOpenVINOShapeInfo(shape_info);
  }
  return option;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto option = BuildOption(FLAGS_device);
  InitAndInfer(FLAGS_model, FLAGS_image, FLAGS_topk, option);
  return 0;
}
