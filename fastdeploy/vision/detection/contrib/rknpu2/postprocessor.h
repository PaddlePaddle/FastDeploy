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

#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/detection/contrib/rknpu2/utils.h"
#include <array>
namespace fastdeploy {
namespace vision {
namespace detection {
/*! @brief Postprocessor object for YOLOv5 serials model.
 */
class FASTDEPLOY_DECL RKYOLOPostprocessor {
 public:
  /** \brief Create a postprocessor instance for YOLOv5 serials model
   */
  RKYOLOPostprocessor();

  /** \brief Process the result of runtime and fill to DetectionResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
           std::vector<DetectionResult>* results);

  /// Set nms_threshold, default 0.45
  void SetNMSThreshold(const float& nms_threshold) {
    nms_threshold_ = nms_threshold;
  }

  /// Set conf_threshold, default 0.25
  void SetConfThreshold(const float& conf_threshold) {
    conf_threshold_ = conf_threshold;
  }

  /// Get conf_threshold, default 0.25
  float GetConfThreshold() const { return conf_threshold_; }

  /// Get nms_threshold, default 0.45
  float GetNMSThreshold() const { return nms_threshold_; }

  // Set height and weight
  void SetHeightAndWeight(int& height, int& width) {
    height_ = height;
    width_ = width;
  }

  // Set pad_hw_values
  void SetPadHWValues(std::vector<std::vector<int>> pad_hw_values) {
    pad_hw_values_ = pad_hw_values;
  }

  // Set scale
  void SetScale(std::vector<float> scale) {
    scale_ = scale;
  }

  // Set Anchor
  void SetAnchor(std::vector<int> anchors, int anchor_per_branch) {
      anchors_ = anchors;
      anchor_per_branch_ = anchor_per_branch;
  }

 private:
  std::vector<int> anchors_ = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                               45, 59, 119, 116, 90, 156, 198, 373, 326};
  int strides_[3] = {8, 16, 32};
  int height_ = 0;
  int width_ = 0;
  int anchor_per_branch_ = 0;

  int ProcessFP16(float *input, int *anchor, int grid_h,
              int grid_w, int stride,
              std::vector<float> &boxes,
              std::vector<float> &boxScores,
              std::vector<int> &classId,
              float threshold);
  // Model
  int QuickSortIndiceInverse(std::vector<float>& input, int left, int right,
                             std::vector<int>& indices);

  // post_process values
  std::vector<std::vector<int>> pad_hw_values_;
  std::vector<float> scale_;
  float nms_threshold_ = 0.45;
  float conf_threshold_ = 0.25;
  int prob_box_size_ = 85;
  int obj_class_num_ = 80;
  int obj_num_bbox_max_size = 200;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
