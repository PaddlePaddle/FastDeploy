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

#include <fstream>

#include "common.h"
#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InferAndCompare(const std::string& model_dir,
                     const std::string& image_file,
                     const std::string& det_result) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto subgraph_file = model_dir + sep + "subgraph.txt";
  fastdeploy::vision::EnableFlyCV();
  fastdeploy::RuntimeOption option;
  option.UseTimVX();
  option.SetLiteSubgraphPartitionPath(subgraph_file);

  auto model = fastdeploy::vision::detection::YOLOv5(
      model_file, params_file, option, fastdeploy::ModelFormat::PADDLE);
  assert(model.Initialized());

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  if (CompareDetResult(res, det_result)) {
    std::cout << model_dir + " run successfully." << std::endl;
  } else {
    std::cerr << model_dir + " run failed." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/quant_model "
                 "path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./yolov5s_quant ./000000014439.jpg "
                 "yolov5_result.txt"
              << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  std::string det_result = argv[3];
  InferAndCompare(model_dir, test_image, det_result);
  return 0;
}
