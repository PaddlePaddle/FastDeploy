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
#include "fastdeploy/vision/detection/ppdet/multiclass_nms.h"
#include "fastdeploy/vision/detection/ppdet/ppdet_decode.h"

namespace fastdeploy {
namespace vision {
namespace detection {
/*! @brief Postprocessor object for PaddleDet serials model.
 */
class FASTDEPLOY_DECL PaddleDetPostprocessor {
 public:
  PaddleDetPostprocessor() = default;

  /** \brief Create a preprocessor instance for PaddleDet serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
   */
  explicit PaddleDetPostprocessor(const std::string& config_file)
      : ppdet_decoder_(config_file) {}

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
           std::vector<DetectionResult>* result);

  /// Apply box decoding and nms step for the outputs for the model.This is
  /// only available for those model exported without box decoding and nms.
  void ApplyDecodeAndNMS(const NMSOption& option = NMSOption()) {
    apply_decode_and_nms_ = true;
    ppdet_decoder_.SetNMSOption(option);
  }

  // Set scale_factor_ value.This is only available for those model exported
  // without box decoding and nms.
  void SetScaleFactor(const std::vector<float>& scale_factor_value) {
    scale_factor_ = scale_factor_value;
  }

 private:
  // for model without decode and nms.
  bool apply_decode_and_nms_ = false;
  bool DecodeAndNMSApplied() const { return apply_decode_and_nms_; }
  bool ProcessUnDecodeResults(const std::vector<FDTensor>& tensors,
                              std::vector<DetectionResult>* results);
  PPDetDecode ppdet_decoder_;
  std::vector<float> scale_factor_{0.0, 0.0};
  std::vector<float> GetScaleFactor() { return scale_factor_; }
  // Process mask tensor for MaskRCNN
  bool ProcessMask(const FDTensor& tensor,
                   std::vector<DetectionResult>* results);
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
