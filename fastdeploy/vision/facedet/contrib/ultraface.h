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
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace facedet {

class FASTDEPLOY_DECL UltraFace : public FastDeployModel {
 public:
  UltraFace(const std::string& model_file, const std::string& params_file = "",
            const RuntimeOption& custom_option = RuntimeOption(),
            const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const {
    return "Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB";
  }

  virtual bool Predict(cv::Mat* im, FaceDetectionResult* result,
                       float conf_threshold = 0.7f,
                       float nms_iou_threshold = 0.3f);

  // tuple of (width, height), default (320, 240)
  std::vector<int> size;

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(std::vector<FDTensor>& infer_result,
                   FaceDetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  bool is_dynamic_input_;
};

}  // namespace facedet
}  // namespace vision
}  // namespace fastdeploy
