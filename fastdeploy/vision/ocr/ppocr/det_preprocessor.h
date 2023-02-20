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
/*! @brief Preprocessor object for DBDetector serials model.
 */
class FASTDEPLOY_DECL DBDetectorPreprocessor : public ProcessorManager {
 public:
  DBDetectorPreprocessor(
      const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
      const std::vector<float>& std = {0.229f, 0.224f, 0.225f});
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch* image_batch, std::vector<FDTensor>* outputs);

  /// Set max_side_len for the detection preprocess, default is 960
  void SetMaxSideLen(int max_side_len) { max_side_len_ = max_side_len; }
  /// Get max_side_len of the detection preprocess
  int GetMaxSideLen() const { return max_side_len_; }

  const std::vector<std::array<int, 4>>* GetBatchImgInfo() {
    return &batch_det_img_info_;
  }

 private:
  bool ResizeImage(FDMat* img, int resize_w, int resize_h, int max_resize_w,
                   int max_resize_h);
  int max_side_len_ = 960;
  std::vector<std::array<int, 4>> batch_det_img_info_;
  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<Pad> pad_op_;
  std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
