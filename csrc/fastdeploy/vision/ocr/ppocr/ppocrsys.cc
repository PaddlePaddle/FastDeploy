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

#include "ppocrsys.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ppocr/utils/ocr_utils.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

namespace fastdeploy {
namespace vision {
namespace ppocr {

PPocrsys::PPocrsys(

    bool use_det, bool use_cls, bool use_rec,

    const std::string& rec_label_path,

    const std::string& det_model_dir, const std::string& cls_model_dir,
    const std::string& rec_model_dir,

    // const RuntimeOption& det_custom_option,
    // const RuntimeOption& cls_custom_option,
    // const RuntimeOption& rec_custom_option,

    const RuntimeOption& ocr_runtime) {
  if (use_det) {
    std::cout << "constuct DB!" << std::endl;
    auto det_model_file = det_model_dir + sep + "inference.pdmodel";
    auto det_params_file = det_model_dir + sep + "inference.pdiparams";
    this->detector =
        new DBDetector(det_model_file, det_params_file, ocr_runtime);
  }

  if (use_cls) {
    std::cout << "constuct CLS!" << std::endl;
    auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
    auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";
    this->classifier =
        new Classifier(cls_model_file, cls_params_file, ocr_runtime);
  }

  if (use_rec) {
    std::cout << "constuct REC!" << std::endl;
    auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
    auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";
    this->recognizer = new Recognizer(rec_label_path, rec_model_file,
                                      rec_params_file, ocr_runtime);
  }
}

void PPocrsys::det(cv::Mat img, std::vector<OCRResult>& ocr_results) {
  std::vector<std::vector<std::vector<int>>> boxes;

  this->detector->Predict(&img, boxes);

  for (int i = 0; i < boxes.size(); i++) {
    OCRResult res;
    res.boxes = boxes[i];
    ocr_results.push_back(res);
  }

  std::cout << "=== Finish DET Prediction ====" << std::endl;
}

void PPocrsys::rec(std::vector<cv::Mat> img_list,
                   std::vector<OCRResult>& ocr_results) {
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

void PPocrsys::cls(std::vector<cv::Mat> img_list,
                   std::vector<OCRResult>& ocr_results) {
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

std::vector<std::vector<OCRResult>> PPocrsys::ocrsys(
    std::vector<cv::Mat> cv_all_imgs, bool use_det, bool use_cls,
    bool use_rec) {
  std::vector<std::vector<OCRResult>> ocr_results;

  if (!use_det) {  //没det
    std::vector<OCRResult> ocr_result;

    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      OCRResult res;
      ocr_result.push_back(res);
    }

    if (use_cls && this->classifier != nullptr) {
      this->cls(cv_all_imgs, ocr_result);
      //摆正图像
      for (int i = 0; i < cv_all_imgs.size(); i++) {
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > this->classifier->cls_thresh) {
          cv::rotate(cv_all_imgs[i], cv_all_imgs[i], 1);
        }
      }
    }

    if (use_rec) {
      this->rec(cv_all_imgs, ocr_result);
    }

    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      std::vector<OCRResult> ocr_result_tmp;
      ocr_result_tmp.push_back(ocr_result[i]);
      ocr_results.push_back(ocr_result_tmp);
    }
  } else {
    //从DET模型开始
    for (int i = 0; i < cv_all_imgs.size(); ++i) {
      std::vector<OCRResult> ocr_result;
      // det
      cv::Mat srcimg = cv_all_imgs[i];
      this->det(srcimg, ocr_result);
      // crop image
      std::vector<cv::Mat> img_list;
      for (int j = 0; j < ocr_result.size(); j++) {
        cv::Mat crop_img;
        crop_img = GetRotateCropImage(srcimg, ocr_result[j].boxes);
        img_list.push_back(crop_img);
      }
      // cls
      if (use_cls && this->classifier != nullptr) {
        // cls模型推理
        this->cls(img_list, ocr_result);

        for (int i = 0; i < img_list.size(); i++) {
          if (ocr_result[i].cls_label % 2 == 1 &&
              ocr_result[i].cls_score > this->classifier->cls_thresh) {
            cv::rotate(img_list[i], img_list[i], 1);
          }
        }
      }
      // rec
      if (use_rec) {
        this->rec(img_list, ocr_result);
      }
      ocr_results.push_back(ocr_result);
    }
  }

  return ocr_results;
}

PPocrsys::~PPocrsys() {
  if (this->detector != nullptr) {
    delete this->detector;
  }
  if (this->classifier != nullptr) {
    delete this->classifier;
  }
  if (this->recognizer != nullptr) {
    delete this->recognizer;
  }
};

}  // namesapce ppocr
}  // namespace vision
}  // namespace fastdeploy