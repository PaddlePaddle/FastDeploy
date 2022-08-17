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
#include "fastdeploy/vision/ppocr/utils/ocr_postprocess_op.h"

#include "./classifier.h"
#include "./dbdetector.h"
#include "./recognizer.h"

namespace fastdeploy {
namespace vision {
namespace ppocr {

class FASTDEPLOY_DECL PPocrsys : public FastDeployModel {
 public:
  PPocrsys(

      bool use_det, bool use_cls, bool use_rec,

      const std::string& rec_label_path,

      const std::string& det_model_dir, const std::string& cls_model_dir,
      const std::string& rec_model_dir,

      const RuntimeOption& ocr_runtime = RuntimeOption());

  DBDetector* detector = nullptr;
  Classifier* classifier = nullptr;
  Recognizer* recognizer = nullptr;

  ~PPocrsys();

  std::vector<std::vector<OCRResult>> ocrsys(
      std::vector<cv::Mat> cv_all_img_names, bool use_det = true,
      bool use_cls = false, bool use_rec = true);

 private:
  void det(cv::Mat img, std::vector<OCRResult>& ocr_results);
  void rec(std::vector<cv::Mat> img_list, std::vector<OCRResult>& ocr_results);
  void cls(std::vector<cv::Mat> img_list, std::vector<OCRResult>& ocr_results);
};

}  // namespace ppocr
}  // namespace vision
}  // namespace fastdeploy
