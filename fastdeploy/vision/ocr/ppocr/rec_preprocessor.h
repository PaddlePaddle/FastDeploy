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
#include "fastdeploy/vision/common/processors/manager.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace ocr {
/*! @brief Preprocessor object for PaddleClas serials model.
 */
class FASTDEPLOY_DECL RecognizerPreprocessor : public ProcessorManager {
 public:
  RecognizerPreprocessor();
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input data list, all the elements are FDMat
   * \param[in] outputs The output tensors which will be fed into runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
           size_t start_index, size_t end_index,
           const std::vector<int>& indices);

  /** \brief Implement the virtual function of ProcessorManager, Apply() is the
   *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
   *  called by users to execute preprocessing
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch* image_batch, std::vector<FDTensor>* outputs);

  /// Set static_shape_infer is true or not. When deploy PP-OCR
  /// on hardware which can not support dynamic input shape very well,
  /// like Huawei Ascned, static_shape_infer needs to to be true.
  void SetStaticShapeInfer(bool static_shape_infer) {
    static_shape_infer_ = static_shape_infer;
  }
  /// Get static_shape_infer of the recognition preprocess
  bool GetStaticShapeInfer() const { return static_shape_infer_; }

  /// Set mean value for the image normalization in recognition preprocess
  void SetMean(const std::vector<float>& mean) { mean_ = mean; }
  /// Get mean value of the image normalization in recognition preprocess
  std::vector<float> GetMean() const { return mean_; }

  /// Set scale value for the image normalization in recognition preprocess
  void SetScale(const std::vector<float>& scale) { scale_ = scale; }
  /// Get scale value of the image normalization in recognition preprocess
  std::vector<float> GetScale() const { return scale_; }

  /// Set is_scale for the image normalization in recognition preprocess
  void SetIsScale(bool is_scale) { is_scale_ = is_scale; }
  /// Get is_scale of the image normalization in recognition preprocess
  bool GetIsScale() const { return is_scale_; }

  /// Set rec_image_shape for the recognition preprocess
  void SetRecImageShape(const std::vector<int>& rec_image_shape) {
    rec_image_shape_ = rec_image_shape;
  }
  /// Get rec_image_shape for the recognition preprocess
  std::vector<int> GetRecImageShape() { return rec_image_shape_; }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize() { disable_permute_ = true; }
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute() { disable_normalize_ = true; }

 private:
  void OcrRecognizerResizeImage(FDMat* mat, float max_wh_ratio,
                              const std::vector<int>& rec_image_shape,
                              bool static_shape_infer);
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  std::vector<int> rec_image_shape_ = {3, 48, 320};
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {0.5f, 0.5f, 0.5f};
  bool is_scale_ = true;
  bool static_shape_infer_ = false;
  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<Pad> pad_op_;
  std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
  std::shared_ptr<Normalize> normalize_op_;
  std::shared_ptr<HWC2CHW> hwc2chw_op_;
  std::shared_ptr<Cast> cast_op_;
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
