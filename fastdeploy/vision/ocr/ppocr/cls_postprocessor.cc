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

#include "fastdeploy/vision/ocr/ppocr/cls_postprocessor.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

ClassifierPostprocessor::ClassifierPostprocessor() { initialized_ = true; }

bool SingleBatchPostprocessor(const float* out_data, const size_t& length,
                              int* cls_label, float* cls_score) {

  *cls_label = std::distance(&out_data[0],
                             std::max_element(&out_data[0], &out_data[length]));

  *cls_score = float(*std::max_element(&out_data[0], &out_data[length]));
  return true;
}

bool ClassifierPostprocessor::Run(const std::vector<FDTensor>& tensors,
                                  std::vector<int32_t>* cls_labels,
                                  std::vector<float>* cls_scores) {
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }
  // Classifier have only 1 output tensor.
  const FDTensor& tensor = tensors[0];

  // For Classifier, the output tensor shape = [batch,2]
  size_t batch = tensor.shape[0];
  size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                             std::multiplies<int>());

  cls_labels->resize(batch);
  cls_scores->resize(batch);
  const float* tensor_data = reinterpret_cast<const float*>(tensor.Data());

  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    if (!SingleBatchPostprocessor(tensor_data, length, &cls_labels->at(i_batch),
                                  &cls_scores->at(i_batch)))
      return false;
    tensor_data = tensor_data + length;
  }

  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
