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

class FASTDEPLOY_DECL NanoDetPlus : public FastDeployModel {
 public:
  NanoDetPlus(const std::string& model_file,
              const std::string& params_file = "",
              const RuntimeOption& custom_option = RuntimeOption(),
              const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "nanodet"; }


  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.35f,
                       float nms_iou_threshold = 0.5f);

  // tuple of input size (width, height), e.g (320, 320)
  std::vector<int> size;
  // padding value, size should be same with Channels
  std::vector<float> padding_value;
  // keep aspect ratio or not when perform resize operation.
  // This option is set as `false` by default in NanoDet-Plus.
  bool keep_ratio;
  // downsample strides for NanoDet-Plus to generate anchors, will
  // take (8, 16, 32, 64) as default values.
  std::vector<int> downsample_strides;
  // for offseting the boxes by classes when using NMS, default 4096.
  float max_wh;
  // reg_max for GFL regression, default 7
  int reg_max;

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, DetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // RangiLyu/nanodet official 'export_onnx.py' script will export static ONNX
  // by default.
  // This value will auto check by fastdeploy after the internal Runtime
  // initialized.
  bool is_dynamic_input_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
