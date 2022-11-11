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

ClassifierPostprocessor::ClassifierPostprocessor() {
  initialized_ = true;
}

bool SingleBatchPostprocessor(const float* out_data, const size_t& length, std::tuple<int, float>* cls_result){

  int label = std::distance(
      &out_data[0], std::max_element(&out_data[0], &out_data[length]));

  float score =
      float(*std::max_element(&out_data[0], &out_data[length]));

  std::get<0>(*cls_result) = label;
  std::get<1>(*cls_result) = score;
  return true;
}

bool ClassifierPostprocessor::Run(const std::vector<FDTensor>& tensors, std::vector<std::tuple<int, float>>* results) {
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }
  // Classifier have only 1 output tensor.
  FDTensor& tensor = tensors[0];

  // For Classifier, the output tensor shape = [batch,2]
  size_t batch = tensor.shape[0];
  size_t length = accumulate(tensor.shape.begin()+1, tensor.shape.end(), 1, multiplies<int>());

  results->resize(batch);
  const float* tensor_data = reinterpret_cast<const float*>(tensor.Data());
 
  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    if(!SingleBatchPostprocessor(tensor_data, length, &results->at(i_batch))) return false;
    tensor_data = tensor_data + length;
  }

  return true;
}

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
