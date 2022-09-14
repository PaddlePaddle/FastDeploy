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

#include "fastdeploy/vision/ocr/ppocr/ppocr_system_v2.h"

namespace fastdeploy {
namespace application {
namespace ocrsystem {

class FASTDEPLOY_DECL PPOCRSystemv3 : public PPOCRSystemv2 {
 public:
  PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Classifier* cls_model,
                fastdeploy::vision::ocr::Recognizer* rec_model) : PPOCRSystemv2(det_model, cls_model, rec_model) {
    // The only difference between v2 and v3
    recognizer_->rec_image_shape[1] = 48;
  }

  PPOCRSystemv3(fastdeploy::vision::ocr::DBDetector* det_model,
                fastdeploy::vision::ocr::Recognizer* rec_model) : PPOCRSystemv2(det_model, rec_model) {
    recognizer_->rec_image_shape[1] = 48;
  }
};

}  // namespace ocrsystem
}  // namespace application
}  // namespace fastdeploy
