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

class FASTDEPLOY_DECL PPOCRSystemv2 : public FastDeployModel {
 public:
  PPOCRSystemv2(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Classifier* cls_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);

  PPOCRSystemv2(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Recognizer* rec_model);

  virtual bool Predict(cv::Mat* img, fastdeploy::vision::OCRResult* result);
  bool Initialized() const override;

 protected:
  fastdeploy::vision::ocr::DBDetector* detector_ = nullptr;
  fastdeploy::vision::ocr::Classifier* classifier_ = nullptr;
  fastdeploy::vision::ocr::Recognizer* recognizer_ = nullptr;

  virtual bool Detect(cv::Mat* img, fastdeploy::vision::OCRResult* result);
  virtual bool Recognize(cv::Mat* img, fastdeploy::vision::OCRResult* result);
  virtual bool Classify(cv::Mat* img, fastdeploy::vision::OCRResult* result);
};

}  // namespace ocrsystem
}  // namespace application
}  // namespace fastdeploy
