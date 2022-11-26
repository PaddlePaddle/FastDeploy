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
#include <iostream>
#include <string>
#include "fastdeploy/vision.h"
#include <sys/time.h>
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
void InferRKYolo(const std::string& model_dir, const std::string& image_file);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./picodet_model_dir ./test.jpeg"
        << std::endl;
    return -1;
  }

  InferRKYolo(argv[1], argv[2]);

  return 0;
}

void InferRKYolo(const std::string& model_dir, const std::string& image_file) {
  struct timeval start_time, stop_time;
  auto model_file = model_dir + "/yolov5s_relu_tk2_RK3588_i8.rknn";

  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();

  auto format = fastdeploy::ModelFormat::RKNN;

  auto model = fastdeploy::vision::detection::RKYOLOV5(
      model_file, option,format);

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  gettimeofday(&start_time, NULL);
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  gettimeofday(&stop_time, NULL);
  printf("infer use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res,0.5);
  cv::imwrite("rkyolo_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./rkyolo_result.jpg" << std::endl;
}