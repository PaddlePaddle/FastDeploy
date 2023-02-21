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

#include "fastdeploy/vision/ocr/ppocr/det_preprocessor.h"

#include "fastdeploy/function/concat.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

std::array<int, 4> OcrDetectorGetInfo(FDMat* img, int max_size_len) {
  int w = img->Width();
  int h = img->Height();

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }
  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);
  resize_h = std::max(int(std::round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(std::round(float(resize_w) / 32) * 32), 32);

  return {w, h, resize_w, resize_h};
  /*
   *ratio_h = float(resize_h) / float(h);
   *ratio_w = float(resize_w) / float(w);
   */
}

DBDetectorPreprocessor::DBDetectorPreprocessor(const std::vector<float>& mean,
                                               const std::vector<float>& std) {
  resize_op_ = CREATE_AND_REGISTER_PROCESSOR(Resize, -1, -1);
  std::vector<float> value = {0, 0, 0};
  pad_op_ = CREATE_AND_REGISTER_PROCESSOR(Pad, 0, 0, 0, 0, value);
  normalize_permute_op_ =
      CREATE_AND_REGISTER_PROCESSOR(NormalizeAndPermute, mean, std, true);
}

bool DBDetectorPreprocessor::ResizeImage(FDMat* img, int resize_w, int resize_h,
                                         int max_resize_w, int max_resize_h) {
  resize_op_->SetWidthAndHeight(resize_w, resize_h);
  (*resize_op_)(img);

  pad_op_->SetPaddingSize(0, max_resize_h - resize_h, 0,
                          max_resize_w - resize_w);
  (*pad_op_)(img);
  return true;
}

bool DBDetectorPreprocessor::Apply(FDMatBatch* image_batch,
                                   std::vector<FDTensor>* outputs) {
  int max_resize_w = 0;
  int max_resize_h = 0;
  batch_det_img_info_.clear();
  batch_det_img_info_.resize(image_batch->mats->size());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    batch_det_img_info_[i] = OcrDetectorGetInfo(mat, max_side_len_);
    max_resize_w = std::max(max_resize_w, batch_det_img_info_[i][2]);
    max_resize_h = std::max(max_resize_h, batch_det_img_info_[i][3]);
  }
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    ResizeImage(mat, batch_det_img_info_[i][2], batch_det_img_info_[i][3],
                max_resize_w, max_resize_h);
  }
  (*normalize_permute_op_)(image_batch);

  outputs->resize(1);
  FDTensor* tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
