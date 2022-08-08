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
#include "fastdeploy/vision/utils/utils.h"

int main() {
  namespace vis = fastdeploy::vision;
  // 0,1 同一个人, 0,2 不同的人
  std::string model_file = "../resources/models/ms1mv3_arcface_r100.onnx";
  std::string face0_path = "../resources/images/face_recognition_0.png";
  std::string face1_path = "../resources/images/face_recognition_1.png";
  std::string face2_path = "../resources/images/face_recognition_2.png";

  auto model = vis::deepinsight::ArcFace(model_file);
  if (!model.Initialized()) {
    std::cerr << "Init Failed! Model: " << model_file << std::endl;
    return -1;
  } else {
    std::cout << "Init Done! Model:" << model_file << std::endl;
  }
  model.EnableDebug();
  // 设置输出l2 normalize后的embedding
  model.l2_normalize = true;

  cv::Mat face0 = cv::imread(face0_path);
  cv::Mat face1 = cv::imread(face1_path);
  cv::Mat face2 = cv::imread(face2_path);

  vis::FaceRecognitionResult res0;
  vis::FaceRecognitionResult res1;
  vis::FaceRecognitionResult res2;
  if ((!model.Predict(&face0, &res0)) || (!model.Predict(&face1, &res1)) ||
      (!model.Predict(&face2, &res2))) {
    std::cerr << "Prediction Failed." << std::endl;
    return -1;
  }
  std::cout << "Prediction Done!" << std::endl;

  // 输出预测框结果
  std::cout << "--- [Face 0]:" << res0.Str();
  std::cout << "--- [Face 1]:" << res1.Str();
  std::cout << "--- [Face 2]:" << res2.Str();

  // 计算余弦相似度
  float cosine01 = vis::utils::CosineSimilarity(res0.embedding, res1.embedding,
                                                model.l2_normalize);
  float cosine02 = vis::utils::CosineSimilarity(res0.embedding, res2.embedding,
                                                model.l2_normalize);
  std::cout << "Detect Done! Cosine 01: " << cosine01
            << ", Cosine 02:" << cosine02 << std::endl;
  return 0;
}
