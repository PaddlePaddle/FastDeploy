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
PPOCRSystemv2::PPOCRSystemv2(fastdeploy::vision::ocr::DBDetector* ocr_det,
                             fastdeploy::vision::ocr::Classifier* ocr_cls,
                             fastdeploy::vision::ocr::Recognizer* ocr_rec)
    : detector(ocr_det), classifier(ocr_cls), recognizer(ocr_rec) {}

void PPOCRSystemv2::Detect(cv::Mat* img,
                           fastdeploy::vision::OCRResult* result) {
  std::vector<std::array<int, 8>> boxes;

  this->detector->Predict(img, &boxes);

  result->boxes = boxes;
}

void PPOCRSystemv2::Recognize(cv::Mat* img,
                              fastdeploy::vision::OCRResult* result) {
  std::tuple<std::string, float> rec_result;
  this->recognizer->rec_image_shape[1] = 32;
  this->recognizer->Predict(img, &rec_result);

  result->text.push_back(std::get<0>(rec_result));
  result->rec_scores.push_back(std::get<1>(rec_result));
}

void PPOCRSystemv2::Classify(cv::Mat* img,
                             fastdeploy::vision::OCRResult* result) {
  std::tuple<int, float> cls_result;

  this->classifier->Predict(img, &cls_result);

  result->cls_labels.push_back(std::get<0>(cls_result));
  result->cls_scores.push_back(std::get<1>(cls_result));
}

bool PPOCRSystemv2::Predict(cv::Mat* img,
                            fastdeploy::vision::OCRResult* result) {
  if (this->detector->initialized == 0) {  //没det
    //输入单张“小图片”给分类器
    if (this->classifier->initialized != 0) {
      this->Classify(img, result);
      //摆正单张图像
      if ((result->cls_labels)[0] % 2 == 1 &&
          (result->cls_scores)[0] > this->classifier->cls_thresh) {
        cv::rotate(*img, *img, 1);
      }
    }
    //输入单张“小图片”给识别器
    if (this->recognizer->initialized != 0) {
      this->Recognize(img, result);
    }

  } else {
    //从DET模型开始
    //一张图,会输出多个“小图片”，送给后续模型
    this->Detect(img, result);
    // crop image
    std::vector<cv::Mat> img_list;

    for (int j = 0; j < (result->boxes).size(); j++) {
      cv::Mat crop_img;
      crop_img =
          fastdeploy::vision::ocr::GetRotateCropImage(*img, (result->boxes)[j]);
      img_list.push_back(crop_img);
    }
    // cls
    if (this->classifier->initialized != 0) {
      for (int i = 0; i < img_list.size(); i++) {
        this->Classify(&img_list[0], result);
      }

      for (int i = 0; i < img_list.size(); i++) {
        if ((result->cls_labels)[i] % 2 == 1 &&
            (result->cls_scores)[i] > this->classifier->cls_thresh) {
          std::cout << "Rotate this image " << std::endl;
          cv::rotate(img_list[i], img_list[i], 1);
        }
      }
    }
    // rec
    if (this->recognizer->initialized != 0) {
      for (int i = 0; i < img_list.size(); i++) {
        this->Recognize(&img_list[i], result);
      }
    }
  }

  return true;
};

}  // namesapce ocrsystem
}  // namespace application
}  // namespace fastdeploy