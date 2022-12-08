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

#include "fastdeploy/vision/ocr/ppocr/cls_preprocessor.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"
#include "fastdeploy/function/concat.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

void OcrClassifierResizeImage(FDMat* mat,
                              const std::vector<int>& cls_image_shape) {
  int img_c = cls_image_shape[0];
  int img_h = cls_image_shape[1];
  int img_w = cls_image_shape[2];

  float ratio = float(mat->Width()) / float(mat->Height());

  int resize_w;
  if (ceilf(img_h * ratio) > img_w)
    resize_w = img_w;
  else
    resize_w = int(ceilf(img_h * ratio));

  Resize::Run(mat, resize_w, img_h);
}

bool ClassifierPreprocessor::Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs) {
  return Run(images, outputs, 0, images->size());
}

bool ClassifierPreprocessor::Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
                                 size_t start_index, size_t end_index) {

  if (images->size() == 0 || start_index <0 || end_index <= start_index || end_index > images->size()) {
    FDERROR << "images->size() or index error. Correct is: 0 <= start_index < end_index <= images->size()" << std::endl;
    return false;
  }

  for (size_t i = start_index; i < end_index; ++i) {
    FDMat* mat = &(images->at(i));
    OcrClassifierResizeImage(mat, cls_image_shape_);
    Normalize::Run(mat, mean_, scale_, is_scale_);
    std::vector<float> value = {0, 0, 0};
    if (mat->Width() < cls_image_shape_[2]) {
      Pad::Run(mat, 0, 0, 0, cls_image_shape_[2] - mat->Width(), value);
    }
    HWC2CHW::Run(mat);
    Cast::Run(mat, "float");
  }
  // Only have 1 output Tensor.
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  size_t tensor_size = end_index - start_index;
  std::vector<FDTensor> tensors(tensor_size); 
  for (size_t i = 0; i < tensor_size; ++i) {
    (*images)[i + start_index].ShareWithTensor(&(tensors[i]));
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
