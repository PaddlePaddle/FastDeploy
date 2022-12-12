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
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"
#include "fastdeploy/function/concat.h"

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

  return {w,h,resize_w,resize_h};
  /*
  *ratio_h = float(resize_h) / float(h);
  *ratio_w = float(resize_w) / float(w);
  */
}
bool OcrDetectorResizeImage(FDMat* img,
                            int resize_w,
                            int resize_h,
                            int max_resize_w,
                            int max_resize_h) {
  Resize::Run(img, resize_w, resize_h);
  std::vector<float> value = {0, 0, 0};
  Pad::Run(img, 0, max_resize_h-resize_h, 0, max_resize_w - resize_w, value);
  return true;
}

bool DBDetectorPreprocessor::Run(std::vector<FDMat>* images,
                                 std::vector<FDTensor>* outputs,
                                 std::vector<std::array<int, 4>>* batch_det_img_info_ptr) {
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0." << std::endl;
    return false;
  }
  int max_resize_w = 0;
  int max_resize_h = 0;
  std::vector<std::array<int, 4>>& batch_det_img_info = *batch_det_img_info_ptr;
  batch_det_img_info.clear();
  batch_det_img_info.resize(images->size());
  for (size_t i = 0; i < images->size(); ++i) {
    FDMat* mat = &(images->at(i));
    batch_det_img_info[i] = OcrDetectorGetInfo(mat,max_side_len_);
    max_resize_w = std::max(max_resize_w,batch_det_img_info[i][2]);
    max_resize_h = std::max(max_resize_h,batch_det_img_info[i][3]);
  }
  for (size_t i = 0; i < images->size(); ++i) {
    FDMat* mat = &(images->at(i));
    OcrDetectorResizeImage(mat, batch_det_img_info[i][2],batch_det_img_info[i][3],max_resize_w,max_resize_h);
    NormalizeAndPermute::Run(mat, mean_, scale_, is_scale_);
    /*
    Normalize::Run(mat, mean_, scale_, is_scale_);
    HWC2CHW::Run(mat);
    Cast::Run(mat, "float");
    */
  }
  // Only have 1 output Tensor.
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(images->size()); 
  for (size_t i = 0; i < images->size(); ++i) {
    (*images)[i].ShareWithTensor(&(tensors[i]));
    tensors[i].ExpandDim(0);
  }
  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
