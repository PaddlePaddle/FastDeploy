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
namespace detection {

class FASTDEPLOY_DECL YOLOv7End2EndORT : public FastDeployModel {
 public:
  YOLOv7End2EndORT(const std::string& model_file,
                   const std::string& params_file = "",
                   const RuntimeOption& custom_option = RuntimeOption(),
                   const ModelFormat& model_format = ModelFormat::ONNX);

  virtual std::string ModelName() const { return "yolov7end2end_ort"; }

  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.25);

  // tuple of (width, height)
  std::vector<int> size;
  // padding value, size should be same with Channels
  std::vector<float> padding_value;
  // only pad to the minimum rectange which height and width is times of stride
  bool is_mini_pad;
  // while is_mini_pad = false and is_no_pad = true, will resize the image to
  // the set size
  bool is_no_pad;
  // if is_scale_up is false, the input image only can be zoom out, the maximum
  // resize scale cannot exceed 1.0
  bool is_scale_up;
  // padding stride, for is_mini_pad
  int stride;

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, DetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold);

  void LetterBox(Mat* mat, const std::vector<int>& size,
                 const std::vector<float>& color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);

  bool is_dynamic_input_;
};
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
