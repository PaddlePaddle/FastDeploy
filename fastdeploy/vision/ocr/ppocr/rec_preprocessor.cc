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

#include "fastdeploy/function/concat.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

void RecognizerPreprocessor::OcrRecognizerResizeImage(FDMat* mat,
                                                      float max_wh_ratio) {
  int img_h = rec_image_shape_[1];
  int img_w = rec_image_shape_[2];
  if (fixed_shape_) {
    // det_image_shape_ is [c,h,w]
    Resize::Run(mat, img_w, img_h);
  }

  if (static_shape_infer_) {
    img_w = int(img_h * max_wh_ratio);
    float ratio = float(mat->Width()) / float(mat->Height());

    int resize_w;
    if (ceilf(img_h * ratio) > img_w) {
      resize_w = img_w;
    } else {
      resize_w = int(ceilf(img_h * ratio));
    }
    Resize::Run(mat, resize_w, img_h);
    Pad::Run(mat, 0, 0, 0, int(img_w - mat->Width()), {127, 127, 127});
  } else {
    if (mat->Width() >= img_w) {
      Resize::Run(mat, img_w, img_h);  // Reszie W to 320
    } else {
      Resize::Run(mat, mat->Width(), img_h);
      Pad::Run(mat, 0, 0, 0, int(img_w - mat->Width()), {127, 127, 127});
      // Pad to 320
    }
  }
}

bool RecognizerPreprocessor::Run(std::vector<FDMat>* images,
                                 std::vector<FDTensor>* outputs) {
  return Run(images, outputs, 0, images->size(), {});
}

bool RecognizerPreprocessor::Run(std::vector<FDMat>* images,
                                 std::vector<FDTensor>* outputs,
                                 size_t start_index, size_t end_index,
                                 const std::vector<int>& indices) {
  if (images->empty() || end_index <= start_index ||
      end_index > images->size()) {
    FDERROR << "images->size() or index error. Correct is: 0 <= start_index < "
               "end_index <= images->size()"
            << std::endl;
    return false;
  }

  int img_h = rec_image_shape_[1];
  int img_w = rec_image_shape_[2];
  float max_wh_ratio = img_w * 1.0 / img_h;
  float ori_wh_ratio;

  for (size_t i = start_index; i < end_index; ++i) {
    size_t real_index = i;
    if (!indices.empty()) {
      real_index = indices[i];
    }
    FDMat* mat = &(images->at(real_index));
    ori_wh_ratio = mat->Width() * 1.0 / mat->Height();
    max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
    OcrRecognizerResizeImage(mat, max_wh_ratio);
    if (!disable_normalize_ && !disable_permute_) {
      NormalizeAndPermute::Run(mat, mean_, scale_, is_scale_);
    } else {
      if (!disable_normalize_) {
        Normalize::Run(mat, mean_, scale_, is_scale_);
      }
      if (!disable_permute_) {
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");
      }
    }
  }
  // Only have 1 output Tensor.
  outputs->resize(1);
  size_t tensor_size = end_index - start_index;
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(tensor_size);
  for (size_t i = 0; i < tensor_size; ++i) {
    size_t real_index = i + start_index;
    if (!indices.empty()) {
      real_index = indices[i + start_index];
    }

    (*images)[real_index].ShareWithTensor(&(tensors[i]));
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
