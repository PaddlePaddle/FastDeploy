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

namespace ocr {
/*! @brief Preprocessor object for DBDetector serials model.
 */
class FASTDEPLOY_DECL DBDetectorPreprocessor {
 public:
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input data list, all the elements are FDMat
   * \param[in] outputs The output tensors which will feed in runtime
   * \param[in] batch_det_img_info_ptr The output of preprocess
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
           std::vector<std::array<int, 4>>* batch_det_img_info_ptr);

  /// Set max_side_len for the detection preprocess, default is 960
  void SetMaxSideLen(int max_side_len) { max_side_len_ = max_side_len; }
  /// Get max_side_len of the detection preprocess
  int GetMaxSideLen() const { return max_side_len_; }

  /// Set mean value for the image normalization in detection preprocess
  void SetMean(const std::vector<float>& mean) { mean_ = mean; }
  /// Get mean value of the image normalization in detection preprocess
  std::vector<float> GetMean() const { return mean_; }

  /// Set scale value for the image normalization in detection preprocess
  void SetScale(const std::vector<float>& scale) { scale_ = scale; }
  /// Get scale value of the image normalization in detection preprocess
  std::vector<float> GetScale() const { return scale_; }

  /// Set is_scale for the image normalization in detection preprocess
  void SetIsScale(bool is_scale) { is_scale_ = is_scale; }
  /// Get is_scale of the image normalization in detection preprocess
  bool GetIsScale() const { return is_scale_; }

 private:
  int max_side_len_ = 960;
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {0.229f, 0.224f, 0.225f};
  bool is_scale_ = true;
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
