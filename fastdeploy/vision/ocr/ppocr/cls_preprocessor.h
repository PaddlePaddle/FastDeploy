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
/*! @brief Preprocessor object for Classifier serials model.
 */
class FASTDEPLOY_DECL ClassifierPreprocessor {
 public:
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input data list, all the elements are FDMat
   * \param[in] outputs The output tensors which will be fed into runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs);
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
           size_t start_index, size_t end_index);

  /// Set mean value for the image normalization in classification preprocess
  void SetMean(const std::vector<float>& mean) { mean_ = mean; }
  /// Get mean value of the image normalization in classification preprocess
  std::vector<float> GetMean() const { return mean_; }

  /// Set scale value for the image normalization in classification preprocess
  void SetScale(const std::vector<float>& scale) { scale_ = scale; }
  /// Get scale value of the image normalization in classification preprocess
  std::vector<float> GetScale() const { return scale_; }

  /// Set is_scale for the image normalization in classification preprocess
  void SetIsScale(bool is_scale) { is_scale_ = is_scale; }
  /// Get is_scale of the image normalization in classification preprocess
  bool GetIsScale() const { return is_scale_; }

  /// Set cls_image_shape for the classification preprocess
  void SetClsImageShape(const std::vector<int>& cls_image_shape) {
    cls_image_shape_ = cls_image_shape;
  }
  /// Get cls_image_shape for the classification preprocess
  std::vector<int> GetClsImageShape() const { return cls_image_shape_; }

 private:
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {0.5f, 0.5f, 0.5f};
  bool is_scale_ = true;
  std::vector<int> cls_image_shape_ = {3, 48, 192};
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
