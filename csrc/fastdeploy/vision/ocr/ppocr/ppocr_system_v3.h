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

#pragma once

#include <vector>

#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

#include "fastdeploy/vision/ocr/ppocr/classifier.h"
#include "fastdeploy/vision/ocr/ppocr/dbdetector.h"
#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace fastdeploy {
namespace application {
namespace ocrsystem {

class FASTDEPLOY_DECL PPOCRSystemv3 : public FastDeployModel {
 public:
  PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* ocr_det = nullptr,
                fastdeploy::vision::ocr::Classifier* ocr_cls = nullptr,
                fastdeploy::vision::ocr::Recognizer* ocr_rec = nullptr);

  fastdeploy::vision::ocr::DBDetector* detector = nullptr;
  fastdeploy::vision::ocr::Classifier* classifier = nullptr;
  fastdeploy::vision::ocr::Recognizer* recognizer = nullptr;

  bool Predict(cv::Mat* img, fastdeploy::vision::OCRResult* result);

 private:
  void Detect(cv::Mat* img, fastdeploy::vision::OCRResult* result);
  void Recognize(cv::Mat* img, fastdeploy::vision::OCRResult* result);
  void Classify(cv::Mat* img, fastdeploy::vision::OCRResult* result);
};

}  // namespace ocrsystem
}  // namespace application
}  // namespace fastdeploy
