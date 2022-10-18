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
#include <unordered_map>

namespace fastdeploy {

namespace vision {

namespace facedet {

class FASTDEPLOY_DECL SCRFD : public FastDeployModel {
 public:
  SCRFD(const std::string& model_file, const std::string& params_file = "",
        const RuntimeOption& custom_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "scrfd"; }
  
  // RKNPU2 can run normalize and hwc2chw on the NPU.
  // This function is used to close normalize and hwc2chw operations in preprocessing.
  void DisableNormalizeAndPermute();

  virtual bool Predict(cv::Mat* im, FaceDetectionResult* result,
                       float conf_threshold = 0.25f,
                       float nms_iou_threshold = 0.4f);

  // tuple of (width, height), default (640, 640)
  std::vector<int> size;
  // downsample strides (namely, steps) for SCRFD to
  // generate anchors, will take (8,16,32) as default values.
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
  // for offseting the boxes by classes when using NMS
  std::vector<int> downsample_strides;
  // landmarks_per_face, default 5 in SCRFD
  int landmarks_per_face;
  // are the outputs of onnx file with key points features or not
  bool use_kps;
  // the upperbond number of boxes processed by nms.
  int max_nms;
  // number anchors of each stride
  unsigned int num_anchors;

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(std::vector<FDTensor>& infer_result,
                   FaceDetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  void GeneratePoints();

  void LetterBox(Mat* mat, const std::vector<int>& size,
                 const std::vector<float>& color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);
  
  // for recording the switch of normalize and hwc2chw
  bool switch_of_nor_and_per = true;  

  bool is_dynamic_input_;

  bool center_points_is_update_;

  typedef struct {
    float cx;
    float cy;
  } SCRFDPoint;

  std::unordered_map<int, std::vector<SCRFDPoint>> center_points_;
};
} // namespace facedet
} // namespace vision
} // namespace fastdeploy
