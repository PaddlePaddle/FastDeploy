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

class FASTDEPLOY_DECL YOLOX : public FastDeployModel {
 public:
  YOLOX(const std::string& model_file, const std::string& params_file = "",
        const RuntimeOption& custom_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "YOLOX"; }

  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.25,
                       float nms_iou_threshold = 0.5);

  // tuple of (width, height)
  std::vector<int> size;
  // padding value, size should be same with Channels
  std::vector<float> padding_value;
  // whether the model_file was exported with decode module. The official
  // YOLOX/tools/export_onnx.py script will export ONNX file without
  // decode module. Please set it 'true' manually if the model file
  // was exported with decode module.
  bool is_decode_exported;
  // downsample strides for YOLOX to generate anchors, will take
  // (8,16,32) as default values, might have stride=64.
  std::vector<int> downsample_strides;
  // for offseting the boxes by classes when using NMS, default 4096.
  float max_wh;

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(FDTensor& infer_result, DetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool PostprocessWithDecode(
      FDTensor& infer_result, DetectionResult* result,
      const std::map<std::string, std::array<float, 2>>& im_info,
      float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // megvii/YOLOX official 'export_onnx.py' script will export static ONNX by
  // default.
  // while is_dynamic_shape if 'false', is_mini_pad will force 'false'. This
  // value will
  // auto check by fastdeploy after the internal Runtime already initialized.
  bool is_dynamic_input_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
