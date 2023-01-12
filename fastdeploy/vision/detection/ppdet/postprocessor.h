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
  explicit PaddleDetPostprocessor(const std::string& config_file);

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
  void ApplyDecodeAndNMS();

  bool DecodeAndNMSApplied();
  // Set scale_factor_ value.This is only available for those model exported
  // without box decoding and nms.
  void SetScaleFactor(float* scale_factor_value);

 private:
  bool apply_decode_and_nms_ = false;

  // for UnDecodeResults
  std::string config_file_;
  std::string arch_;
  std::vector<float> fpn_stride_{8, 16, 32, 64};
  std::vector<float> scale_factor_{1.0, 1.0};
  std::vector<float> im_shape_{416, 416};
  float score_threshold_ = 0.5;
  float nms_threshold_ = 0.5;
  std::vector<float> GetScaleFactor();
  bool ReadPostprocessConfigFromYaml();
  bool ProcessUnDecodeResults(const std::vector<FDTensor>& tensors,
                              std::vector<DetectionResult>* results);
  void DisPred2Bbox(const float*& dfl_det, int label, float score, int x, int y,
                    int stride, int reg_max,
                    fastdeploy::vision::DetectionResult* results);
  void PicoDetPostProcess(DetectionResult* results,
                          const std::vector<FDTensor>& outs, int reg_max,
                          int num_class);

  // Process mask tensor for MaskRCNN
  bool ProcessMask(const FDTensor& tensor,
                   std::vector<DetectionResult>* results);
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
