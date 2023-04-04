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

void RKNPU2Infer(const std::string& model_file, const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();

  auto format = fastdeploy::ModelFormat::RKNN;

  auto model =
      fastdeploy::vision::detection::RKYOLOV5(model_file, option, format);

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  fastdeploy::TimeCounter tc;
  tc.Start();
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  tc.End();
  tc.PrintInfo("RKYOLOV5 in RKNN");
  std::cout << res.Str() << std::endl;
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./picodet_model_dir ./test.jpeg"
        << std::endl;
    return -1;
  }

  RKNPU2Infer(argv[1], argv[2]);

  return 0;
}
