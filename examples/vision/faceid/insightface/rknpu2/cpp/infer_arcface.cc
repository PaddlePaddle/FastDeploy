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

void CpuInfer(const std::string& model_file,
              const std::vector<std::string>& image_file) {
  auto model = fastdeploy::vision::faceid::ArcFace(model_file, "");

  cv::Mat face0 = cv::imread(image_file[0]);
  fastdeploy::vision::FaceRecognitionResult res0;
  if (!model.Predict(face0, &res0)) {
    std::cerr << "Prediction Failed." << std::endl;
  }

  cv::Mat face1 = cv::imread(image_file[1]);
  fastdeploy::vision::FaceRecognitionResult res1;
  if (!model.Predict(face1, &res1)) {
    std::cerr << "Prediction Failed." << std::endl;
  }

  cv::Mat face2 = cv::imread(image_file[2]);
  fastdeploy::vision::FaceRecognitionResult res2;
  if (!model.Predict(face2, &res2)) {
    std::cerr << "Prediction Failed." << std::endl;
    return;
  }

  std::cout << "Prediction Done!" << std::endl;

  std::cout << "--- [Face 0]:" << res0.Str();
  std::cout << "--- [Face 1]:" << res1.Str();
  std::cout << "--- [Face 2]:" << res2.Str();

  float cosine01 = fastdeploy::vision::utils::CosineSimilarity(
      res0.embedding, res1.embedding,
      model.GetPostprocessor().GetL2Normalize());
  float cosine02 = fastdeploy::vision::utils::CosineSimilarity(
      res0.embedding, res2.embedding,
      model.GetPostprocessor().GetL2Normalize());
  std::cout << "Detect Done! Cosine 01: " << cosine01
            << ", Cosine 02:" << cosine02 << std::endl;
}

void RKNPUInfer(const std::string& model_file,
                const std::vector<std::string>& image_file) {
  std::string params_file;
  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();
  auto format = fastdeploy::ModelFormat::RKNN;
  auto model = fastdeploy::vision::faceid::ArcFace(model_file, params_file,
                                                   option, format);
  model.GetPreprocessor().DisableNormalize();
  model.GetPreprocessor().DisablePermute();

  cv::Mat face0 = cv::imread(image_file[0]);
  fastdeploy::vision::FaceRecognitionResult res0;
  if (!model.Predict(face0, &res0)) {
    std::cerr << "Prediction Failed." << std::endl;
    return;
  }

  cv::Mat face1 = cv::imread(image_file[1]);
  fastdeploy::vision::FaceRecognitionResult res1;
  if (!model.Predict(face1, &res1)) {
    std::cerr << "Prediction Failed." << std::endl;
    return;
  }

  cv::Mat face2 = cv::imread(image_file[2]);
  fastdeploy::vision::FaceRecognitionResult res2;
  if (!model.Predict(face2, &res2)) {
    std::cerr << "Prediction Failed." << std::endl;
    return;
  }

  std::cout << "Prediction Done!" << std::endl;

  std::cout << "--- [Face 0]:" << res0.Str();
  std::cout << "--- [Face 1]:" << res1.Str();
  std::cout << "--- [Face 2]:" << res2.Str();

  float cosine01 = fastdeploy::vision::utils::CosineSimilarity(
      res0.embedding, res1.embedding,
      model.GetPostprocessor().GetL2Normalize());
  float cosine02 = fastdeploy::vision::utils::CosineSimilarity(
      res0.embedding, res2.embedding,
      model.GetPostprocessor().GetL2Normalize());
  std::cout << "Detect Done! Cosine 01: " << cosine01
            << ", Cosine 02:" << cosine02 << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_arcface_demo ms1mv3_arcface_r100.onnx "
                 "face_0.jpg face_1.jpg face_2.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, "
                 "0: run with cpu; 1: run with rknpu2."
              << std::endl;
    return -1;
  }

  std::vector<std::string> image_files = {argv[2], argv[3], argv[4]};
  if (std::atoi(argv[5]) == 0) {
    CpuInfer(argv[1], image_files);
  } else if (std::atoi(argv[5]) == 1) {
    RKNPUInfer(argv[1], image_files);
  }
  return 0;
}
