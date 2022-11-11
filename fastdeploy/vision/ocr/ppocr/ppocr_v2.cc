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
  recognizer_->preprocessor_.rec_image_shape_[1] = 32;
}

PPOCRv2::PPOCRv2(fastdeploy::vision::ocr::DBDetector* det_model,
                             fastdeploy::vision::ocr::Recognizer* rec_model)
    : detector_(det_model), recognizer_(rec_model) {
  recognizer_->preprocessor_.rec_image_shape_[1] = 32;
}

bool PPOCRv2::Initialized() const {
  
  if (detector_ != nullptr && !detector_->Initialized()){
    return false;
  }

  if (classifier_ != nullptr && !classifier_->Initialized()){
    return false;
  }

  if (recognizer_ != nullptr && !recognizer_->Initialized()){
    return false;
  }
  return true; 
}

// This function is obsolete
bool PPOCRv2::Detect(cv::Mat* img,
                           fastdeploy::vision::OCRResult* result) {
  fastdeploy::vision::OCRBatchResult batch_result;
  BatchDetect({*img},&batch_result);
  result->boxes = std::move(batch_result.batch_boxes[0]);
  vision::ocr::SortBoxes(&(result->boxes));
  return true;
}

// This function is obsolete
bool PPOCRv2::Recognize(cv::Mat* img,
                              fastdeploy::vision::OCRResult* result) {
  std::vector<std::string> text;
  std::vector<float> rec_scores;
  BatchRecognize({*img}, &text, &rec_scores);

  result->text.emplace_back(std::move(text[0]));
  result->rec_scores.emplace_back(std::move(rec_scores[0]));
  return true;
}
// This function is obsolete
bool PPOCRv2::Classify(cv::Mat* img,
                             fastdeploy::vision::OCRResult* result) {
  std::vector<int32_t> cls_labels;
  std::vector<float> cls_scores;
  BatchClassify({*img}, &cls_labels, &cls_scores);

  result->cls_labels.emplace_back(std::move(cls_labels[0]));
  result->cls_scores.emplace_back(std::move(cls_scores[0]));
  return true;
}

bool PPOCRv2::Predict(cv::Mat* img,
                            fastdeploy::vision::OCRResult* result) {
  result->Clear();
  fastdeploy::vision::OCRBatchResult batch_result;
  BatchPredict({img},&batch_result);
  
  result->boxes = std::move(batch_result.batch_boxes[0]);
  result->text = std::move(batch_result.batch_text[0]);
  result->rec_scores = std::move(batch_result.batch_rec_scores[0]);
  result->cls_scores = std::move(batch_result.batch_cls_scores[0]);
  result->cls_labels = std::move(batch_result.batch_cls_labels[0]);
  return true;
};

bool PPOCRv2::BatchDetect(const std::vector<cv::Mat>& images, fastdeploy::vision::OCRBatchResult* batch_result){
  if (!detector_->BatchPredict(images, &(batch_result->batch_boxes))) {
    FDERROR << "There's error while detecting image in PPOCR." << std::endl;
    return false;
  }
  for(int i_batch = 0; i_batch < batch_result->batch_boxes.size(); ++i_batch){
    vision::ocr::SortBoxes(&batch_result->batch_boxes[i_batch]);
  }
  return true;
}

// For Recognize, batch = the boxes number of 1 image for now.
// Merge multiple images in Recognize Model will support later.
// Additional information is required to record which boxes belong to 1 image.
bool PPOCRv2::BatchRecognize(const std::vector<cv::Mat>& images, std::vector<std::string>* text, std::vector<float>* rec_scores){
  std::vector<std::tuple<std::string, float>> batch_rec_result;
  if (!recognizer_->BatchPredict(images, &batch_rec_result)) {
    FDERROR << "There's error while recognizing image in PPOCR." << std::endl;
    return false;
  }
  text->resize(batch_rec_result.size());
  rec_scores->resize(batch_rec_result.size());
  for(i_batch = 0; i_batch < batch_rec_result.size(); ++i_batch){
    text[i_batch] = std::get<0>(batch_rec_result[i_batch]);
    rec_scores[i_batch] = std::get<1>(batch_rec_result[i_batch]);
  }
  return true;
}

// For Classify, batch = the boxes number of 1 image for now.
// Merge multiple images in Recognize Model will support later.
// Additional information is required to record which boxes belong to 1 image.
bool PPOCRv2::BatchClassify(const std::vector<cv::Mat>& images, std::vector<int32_t>* cls_labels, std::vector<float>* cls_scores){
  std::vector<std::tuple<int, float>> batch_cls_result;
  if (!classifier_->BatchPredict(images, &batch_cls_result)) {
    FDERROR << "There's error while recognizing image in PPOCR." << std::endl;
    return false;
  }
  cls_labels->resize(batch_cls_result.size());
  cls_scores->resize(batch_cls_result.size());
  for(i_batch = 0; i_batch < batch_cls_result.size(); ++i_batch){
    cls_labels[i_batch] = std::get<0>(batch_cls_result[i_batch]);
    cls_scores[i_batch] = std::get<1>(batch_cls_result[i_batch]);
  }
  return true;
}

bool PPOCRv2::BatchPredict(const std::vector<cv::Mat>& images, fastdeploy::vision::OCRBatchResult* batch_result){
  batch_result->Clear();
  batch_result->batch_boxes.resize(images.size());
  batch_result->batch_text.resize(images.size());
  batch_result->batch_rec_scores.resize(images.size());
  batch_result->batch_cls_scores.resize(images.size());
  batch_result->batch_cls_labels.resize(images.size());

  if (nullptr != detector_ && !BatchDetect(images, batch_result)) {
    FDERROR << "Failed to detect image." << std::endl;
    return false;
  }
  for(int i_batch = 0; i_batch < images.size(); ++i_batch){
    // Get croped images by detection result
    const std::vector<std::array<int, 8>>& boxes = batch_result->batch_boxes[i_batch];
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
    std::vector<int32_t>* cls_labels_ptr = &batch_result->batch_cls_labels[i_batch];
    std::vector<float>* cls_scores_ptr = &batch_result->batch_cls_scores[i_batch];

    std::vector<std::string>* text_ptr = &batch_result->batch_text[i_batch];
    std::vector<float>* rec_scores_ptr = &batch_result->batch_rec_scores[i_batch];

    if (nullptr != classifier_ && !BatchClassify(image_list, cls_labels_ptr, cls_scores_ptr)) {
      FDERROR << "Failed to classify croped images." << std::endl;
      return false;
    }else{
      for (size_t i_img = 0; i_img < image_list.size(); ++i_img) {
        if(cls_labels_ptr[i_img] % 2 == 1 && cls_scores_ptr[i_img] > classifier_->postprocessor_.cls_thresh_){
          cv::rotate(image_list[i_img], image_list[i_img], 1);
        }
      }
    }

    if (nullptr != recognizer_ && !BatchRecognize(image_list, text_ptr, rec_scores_ptr)) {
      FDERROR << "Failed to recgnize croped image of index." << std::endl;
      return false;
    }
  }
  return true;
}

}  // namesapce pipeline
}  // namespace fastdeploy
