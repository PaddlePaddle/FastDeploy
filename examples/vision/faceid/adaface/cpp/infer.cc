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

void CpuInfer(const std::string &model_file, const std::string &params_file,
              const std::vector<std::string> &image_file) {
  auto option = fastdeploy::RuntimeOption();
  auto model = fastdeploy::vision::faceid::AdaFace(model_file, params_file);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  cv::Mat face0 = cv::imread(image_file[0]);
  cv::Mat face1 = cv::imread(image_file[1]);
  cv::Mat face2 = cv::imread(image_file[2]);

  fastdeploy::vision::FaceRecognitionResult res0;
  fastdeploy::vision::FaceRecognitionResult res1;
  fastdeploy::vision::FaceRecognitionResult res2;

  if ((!model.Predict(face0, &res0)) || (!model.Predict(face1, &res1)) ||
      (!model.Predict(face2, &res2))) {
    std::cerr << "Prediction Failed." << std::endl;
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

void KunlunXinInfer(const std::string &model_file, const std::string &params_file,
              const std::vector<std::string> &image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseKunlunXin();
  auto model = fastdeploy::vision::faceid::AdaFace(model_file, params_file);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  cv::Mat face0 = cv::imread(image_file[0]);
  cv::Mat face1 = cv::imread(image_file[1]);
  cv::Mat face2 = cv::imread(image_file[2]);

  fastdeploy::vision::FaceRecognitionResult res0;
  fastdeploy::vision::FaceRecognitionResult res1;
  fastdeploy::vision::FaceRecognitionResult res2;

  if ((!model.Predict(face0, &res0)) || (!model.Predict(face1, &res1)) ||
      (!model.Predict(face2, &res2))) {
    std::cerr << "Prediction Failed." << std::endl;
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

void GpuInfer(const std::string &model_file, const std::string &params_file,
              const std::vector<std::string> &image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model =
      fastdeploy::vision::faceid::AdaFace(model_file, params_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  cv::Mat face0 = cv::imread(image_file[0]);
  cv::Mat face1 = cv::imread(image_file[1]);
  cv::Mat face2 = cv::imread(image_file[2]);

  fastdeploy::vision::FaceRecognitionResult res0;
  fastdeploy::vision::FaceRecognitionResult res1;
  fastdeploy::vision::FaceRecognitionResult res2;

  if ((!model.Predict(face0, &res0)) || (!model.Predict(face1, &res1)) ||
      (!model.Predict(face2, &res2))) {
    std::cerr << "Prediction Failed." << std::endl;
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

void TrtInfer(const std::string &model_file, const std::string &params_file,
              const std::vector<std::string> &image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.SetTrtInputShape("data", {1, 3, 112, 112});
  auto model =
      fastdeploy::vision::faceid::AdaFace(model_file, params_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  cv::Mat face0 = cv::imread(image_file[0]);
  cv::Mat face1 = cv::imread(image_file[1]);
  cv::Mat face2 = cv::imread(image_file[2]);

  fastdeploy::vision::FaceRecognitionResult res0;
  fastdeploy::vision::FaceRecognitionResult res1;
  fastdeploy::vision::FaceRecognitionResult res2;

  if ((!model.Predict(face0, &res0)) || (!model.Predict(face1, &res1)) ||
      (!model.Predict(face2, &res2))) {
    std::cerr << "Prediction Failed." << std::endl;
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

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_demo mobilefacenet_adaface.pdmodel "
                 "mobilefacenet_adaface.pdiparams "
                 "test_lite_focal_AdaFace_0.JPG test_lite_focal_AdaFace_1.JPG "
                 "test_lite_focal_AdaFace_2.JPG 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend; 3: run with kunlunxin."
              << std::endl;
    return -1;
  }

  std::vector<std::string> image_files = {argv[3], argv[4], argv[5]};
  if (std::atoi(argv[6]) == 0) {
    std::cout << "use CpuInfer" << std::endl;
    CpuInfer(argv[1], argv[2], image_files);
  } else if (std::atoi(argv[6]) == 1) {
    GpuInfer(argv[1], argv[2], image_files);
  } else if (std::atoi(argv[6]) == 2) {
    TrtInfer(argv[1], argv[2], image_files);
  } else if (std::atoi(argv[6]) == 3) {
    KunlunXinInfer(argv[1], argv[2], image_files);
  }
  return 0;
}
