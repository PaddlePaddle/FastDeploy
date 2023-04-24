// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "fastdeploy/vision/ocr/ppocr/structurev2_layout_preprocessor.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

StructureV2LayoutPreprocessor::StructureV2LayoutPreprocessor() {
  // default width(608) and height(900)
  resize_op_ = std::make_shared<Resize>(image_shape_[2], image_shape_[1]);
  normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
      std::vector<float>({0.485f, 0.456f, 0.406f}),
      std::vector<float>({0.229f, 0.224f, 0.225f}), true);
}

bool StructureV2LayoutPreprocessor::Apply(FDMatBatch* image_batch, 
                                          std::vector<FDTensor>* outputs) {
  return true;                                          
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
