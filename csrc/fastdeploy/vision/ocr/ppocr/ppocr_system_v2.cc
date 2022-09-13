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

#include "fastdeploy/vision/ocr/ppocr/ppocr_system_v2.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace application {
namespace ocrsystem {
PPOCRSystemv2::PPOCRSystemv2(fastdeploy::vision::ocr::DBDetector* det_model,
                             fastdeploy::vision::ocr::Classifier* cls_model,
                             fastdeploy::vision::ocr::Recognizer* rec_model)
    : detector_(det_model), classifier_(cls_model), recognizer_(rec_model) {
  FDERROR << "???????1" << std::endl;
  recognizer_->rec_image_shape[1] = 32;
}

PPOCRSystemv2::PPOCRSystemv2(fastdeploy::vision::ocr::DBDetector* det_model,
                             fastdeploy::vision::ocr::Recognizer* rec_model)
    : detector_(det_model), recognizer_(rec_model) {
  FDERROR << "???????2" << std::endl;
  recognizer_->rec_image_shape[1] = 32;
}

bool PPOCRSystemv2::Detect(cv::Mat* img,
                           fastdeploy::vision::OCRResult* result) {
  if (!detector_->Predict(img, &(result->boxes))) {
    FDERROR << "There's error while detecting image in PPOCRSystem." << std::endl;
    return false;
  }
  vision::ocr::SortBoxes(result);
  return true;
}

bool PPOCRSystemv2::Recognize(cv::Mat* img,
                              fastdeploy::vision::OCRResult* result) {
  std::tuple<std::string, float> rec_result;
  if (!recognizer_->Predict(img, &rec_result)) {
    FDERROR << "There's error while recognizing image in PPOCRSystem." << std::endl;
    return false;
  }

  result->text.push_back(std::get<0>(rec_result));
  result->rec_scores.push_back(std::get<1>(rec_result));
  return true;
}

bool PPOCRSystemv2::Classify(cv::Mat* img,
                             fastdeploy::vision::OCRResult* result) {
  std::tuple<int, float> cls_result;

  if (!classifier_->Predict(img, &cls_result)) {
    FDERROR << "There's error while classifying image in PPOCRSystem." << std::endl;
    return false;
  }

  result->cls_labels.push_back(std::get<0>(cls_result));
  result->cls_scores.push_back(std::get<1>(cls_result));
  return true;
}

bool PPOCRSystemv2::Predict(cv::Mat* img,
                            fastdeploy::vision::OCRResult* result) {
  result->Clear();
  if (nullptr != detector_ && !Detect(img, result)) {
    FDERROR << "Failed to detect image." << std::endl;
    return false;
  }

  // Get croped images by detection result
  std::vector<cv::Mat> image_list;
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    auto crop_im = vision::ocr::GetRotateCropImage(*img, (result->boxes)[i]);
    image_list.push_back(crop_im);
  }
  if (result->boxes.size() == 0) {
    image_list.push_back(*img);
  }

  for (size_t i = 0; i < image_list.size(); ++i) {
    if (nullptr != classifier_ && !Classify(&(image_list[i]), result)) {
      FDERROR << "Failed to classify croped image of index " << i << "." << std::endl;
      return false;
    }
    if (nullptr != classifier_ && result->cls_labels[i] % 2 == 1 && result->cls_scores[i] > classifier_->cls_thresh) {
      cv::rotate(image_list[i], image_list[i], 1);
    }

    if (nullptr != recognizer_ && !Recognize(&(image_list[i]), result)) {
      FDERROR << "Failed to recgnize croped image of index " << i << "." << std::endl;
      return false;
    }
  }
  return true;
};

}  // namesapce ocrsystem
}  // namespace application
}  // namespace fastdeploy
