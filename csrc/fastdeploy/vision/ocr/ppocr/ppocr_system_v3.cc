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

#include "fastdeploy/vision/ocr/ppocr/ppocr_system_v3.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace application {
namespace ocrsystem {
PPOCRSystemv3::PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* ocr_det,
                             fastdeploy::vision::ocr::Classifier* ocr_cls,
                             fastdeploy::vision::ocr::Recognizer* ocr_rec) {
  this->detector = ocr_det;
  this->classifier = ocr_cls;
  this->recognizer = ocr_rec;
}

void PPOCRSystemv3::Detect(
    cv::Mat img, std::vector<fastdeploy::vision::OCRResult>& ocr_results) {
  std::vector<std::vector<std::vector<int>>> boxes;

  this->detector->Predict(&img, boxes);

  for (int i = 0; i < boxes.size(); i++) {
    fastdeploy::vision::OCRResult res;
    res.boxes = boxes[i];
    ocr_results.push_back(res);
  }

  std::cout << "=== Finish DET Prediction ====" << std::endl;
}

void PPOCRSystemv3::Recognize(
    std::vector<cv::Mat> img_list,
    std::vector<fastdeploy::vision::OCRResult>& ocr_results) {
  std::vector<std::string> rec_texts(img_list.size(), "");
  std::vector<float> rec_text_scores(img_list.size(), 0);

  this->recognizer->Predict(img_list, rec_texts, rec_text_scores);

  // output rec results
  for (int i = 0; i < rec_texts.size(); i++) {
    ocr_results[i].text = rec_texts[i];
    ocr_results[i].score = rec_text_scores[i];
  }
  std::cout << "=== Finish REC Prediction ====" << std::endl;
}

void PPOCRSystemv3::Classify(
    std::vector<cv::Mat> img_list,
    std::vector<fastdeploy::vision::OCRResult>& ocr_results) {
  std::vector<int> cls_labels(img_list.size(), 0);
  std::vector<float> cls_scores(img_list.size(), 0);

  this->classifier->Predict(img_list, cls_labels, cls_scores);
  // output cls results
  for (int i = 0; i < cls_labels.size(); i++) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
  std::cout << "=== Finish CLS Prediction ====" << std::endl;
}

std::vector<std::vector<fastdeploy::vision::OCRResult>> PPOCRSystemv3::Predict(
    std::vector<cv::Mat> cv_all_imgs) {
  std::vector<std::vector<fastdeploy::vision::OCRResult>> ocr_results;

  if (this->detector == nullptr) {  //没det
    std::vector<fastdeploy::vision::OCRResult> ocr_result;

    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      fastdeploy::vision::OCRResult res;
      ocr_result.push_back(res);
    }

    if (this->classifier != nullptr) {
      this->Classify(cv_all_imgs, ocr_result);
      //摆正图像
      for (int i = 0; i < cv_all_imgs.size(); i++) {
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > this->classifier->cls_thresh) {
          cv::rotate(cv_all_imgs[i], cv_all_imgs[i], 1);
        }
      }
    }

    if (this->recognizer != nullptr) {
      this->Recognize(cv_all_imgs, ocr_result);
    }

    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      std::vector<fastdeploy::vision::OCRResult> ocr_result_tmp;
      ocr_result_tmp.push_back(ocr_result[i]);
      ocr_results.push_back(ocr_result_tmp);
    }
  } else {
    //从DET模型开始
    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      std::vector<fastdeploy::vision::OCRResult> ocr_result;
      // det
      cv::Mat srcimg = cv_all_imgs[i];
      this->Detect(srcimg, ocr_result);
      // crop image
      std::vector<cv::Mat> img_list;
      for (int j = 0; j < ocr_result.size(); j++) {
        cv::Mat crop_img;
        crop_img = fastdeploy::vision::ocr::GetRotateCropImage(
            srcimg, ocr_result[j].boxes);
        img_list.push_back(crop_img);
      }
      // cls
      if (this->classifier != nullptr) {
        // cls模型推理
        this->Classify(img_list, ocr_result);

        for (int i = 0; i < img_list.size(); i++) {
          if (ocr_result[i].cls_label % 2 == 1 &&
              ocr_result[i].cls_score > this->classifier->cls_thresh) {
            std::cout << "Rotate this image " << std::endl;
            cv::rotate(img_list[i], img_list[i], 1);
          }
        }
      }
      // rec
      if (this->recognizer != nullptr) {
        this->Recognize(img_list, ocr_result);
      }
      ocr_results.push_back(ocr_result);
    }
  }

  return ocr_results;
};

}  // namesapce ocrsystem
}  // namespace application
}  // namespace fastdeploy