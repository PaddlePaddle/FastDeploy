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

#include "fastdeploy/vision/ocr/ppocr/ppocr_v2.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace pipeline {
PPOCRv2::PPOCRv2(fastdeploy::vision::ocr::DBDetector* det_model,
                             fastdeploy::vision::ocr::Classifier* cls_model,
                             fastdeploy::vision::ocr::Recognizer* rec_model)
    : detector_(det_model), classifier_(cls_model), recognizer_(rec_model) {
  Initialized();
  recognizer_->preprocessor_.rec_image_shape_[1] = 32;
}

PPOCRv2::PPOCRv2(fastdeploy::vision::ocr::DBDetector* det_model,
                             fastdeploy::vision::ocr::Recognizer* rec_model)
    : detector_(det_model), recognizer_(rec_model) {
  Initialized();
  recognizer_->preprocessor_.rec_image_shape_[1] = 32;
}

bool PPOCRv2::Initialized() const {
  
  if (detector_ != nullptr && !detector_->Initialized()) {
    return false;
  }

  if (classifier_ != nullptr && !classifier_->Initialized()) {
    return false;
  }

  if (recognizer_ != nullptr && !recognizer_->Initialized()) {
    return false;
  }
  return true; 
}

bool PPOCRv2::Predict(cv::Mat* img,
                            fastdeploy::vision::OCRResult* result) {
  BatchPredict({*img},&{*result});
  return true;
};

bool PPOCRv2::BatchPredict(const std::vector<cv::Mat>& images,
                           std::vector<fastdeploy::vision::OCRResult>* batch_result) {
  batch_result->clear();
  batch_result->resize(images.size());
  std::vector<std::vector<std::array<int, 8>>> batch_boxes(images.size());

  if (!detector_->BatchPredict(images, &batch_boxes)) {
    FDERROR << "There's error while detecting image in PPOCR." << std::endl;
    return false;
  }
  for(int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
    vision::ocr::SortBoxes(&(batch_boxes[i_batch]));
    (*batch_result)[i_batch].boxes = batch_boxes[i_batch];
  }
  
  
  for(int i_batch = 0; i_batch < images.size(); ++i_batch) {
    fastdeploy::vision::OCRResult& ocr_result = (*batch_result)[i_batch];
    // Get croped images by detection result
    const std::vector<std::array<int, 8>>& boxes = ocr_result.boxes;
    const cv::Mat& img = images[i_batch];
    std::vector<cv::Mat> image_list;
    if (boxes.size() == 0) {
      image_list.emplace_back(img);
    }else{
      image_list.resize(boxes.size());
      for (size_t i_box = 0; i_box < boxes.size(); ++i_box) {
        image_list[i_box] = vision::ocr::GetRotateCropImage(img, boxes[i_box]);
      }
    }
    std::vector<int32_t>* cls_labels_ptr = &ocr_result.cls_labels;
    std::vector<float>* cls_scores_ptr = &ocr_result.cls_scores;

    std::vector<std::string>* text_ptr = &ocr_result.text;
    std::vector<float>* rec_scores_ptr = &ocr_result.rec_scores;

    if (!classifier_->BatchPredict(images, cls_labels_ptr, cls_scores_ptr)) {
      FDERROR << "There's error while recognizing image in PPOCR." << std::endl;
      return false;
    }else{
      for (size_t i_img = 0; i_img < image_list.size(); ++i_img) {
        if(*cls_labels_ptr[i_img] % 2 == 1 && *cls_scores_ptr[i_img] > classifier_->postprocessor_.cls_thresh_) {
          cv::rotate(image_list[i_img], image_list[i_img], 1);
        }
      }
    }

    if (!recognizer_->BatchPredict(images, text_ptr, rec_scores_ptr)) {
      FDERROR << "There's error while recognizing image in PPOCR." << std::endl;
      return false;
    }
  }
  return true;
}

}  // namesapce pipeline
}  // namespace fastdeploy
