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

#include "fastdeploy/vision/ocr/ppocr/rec_preprocessor.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"
#include "fastdeploy/function/concat.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

RecognizerPreprocessor::RecognizerPreprocessor() {
  initialized_ = true;
}

void OcrRecognizerResizeImage(FDMat* mat, float max_wh_ratio,
                              const std::vector<int>& rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(imgH * max_wh_ratio);

  float ratio = float(mat->Width()) / float(mat->Height());
  int resize_w;
  if (ceilf(imgH * ratio) > imgW) {
    resize_w = imgW;
  }else{
    resize_w = int(ceilf(imgH * ratio));
  }
  Resize::Run(mat, resize_w, imgH);

  std::vector<float> value = {0, 0, 0};
  Pad::Run(mat, 0, 0, 0, int(imgW - mat->Width()), value);
}

bool RecognizerPreprocessor::Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs) {
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0." << std::endl;
    return false;
  }

  int imgH = rec_image_shape_[1];
  int imgW = rec_image_shape_[2];
  float max_wh_ratio = imgW * 1.0 / imgH;
  float ori_wh_ratio;
  
  for (size_t i = 0; i < images->size(); ++i) {
    FDMat* mat = &(images->at(i));
    ori_wh_ratio = mat->Width() * 1.0 / mat->Height();
    max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
  }

  for (size_t i = 0; i < images->size(); ++i) {
    FDMat* mat = &(images->at(i));
    OcrRecognizerResizeImage(mat, max_wh_ratio, rec_image_shape_);
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
