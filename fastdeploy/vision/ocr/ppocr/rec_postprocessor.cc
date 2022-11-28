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

#include "fastdeploy/vision/ocr/ppocr/rec_postprocessor.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

std::vector<std::string> ReadDict(const std::string& path) {
  std::ifstream in(path);
  FDASSERT(in, "Cannot open file %s to read.", path.c_str());
  std::string line;
  std::vector<std::string> m_vec;
  while (getline(in, line)) {
    m_vec.push_back(line);
  }
  m_vec.insert(m_vec.begin(), "#");  // blank char for ctc
  m_vec.push_back(" ");
  return m_vec;
}

RecognizerPostprocessor::RecognizerPostprocessor() { initialized_ = true; }

RecognizerPostprocessor::RecognizerPostprocessor(
    const std::string& label_path) {
  // init label_lsit
  label_list_ = ReadDict(label_path);
  initialized_ = true;
}

bool RecognizerPostprocessor::SingleBatchPostprocessor(
    const float* out_data, const std::vector<int64_t>& output_shape,
    std::string* text, float* rec_score) {
  std::string& str_res = *text;
  float& score = *rec_score;
  score = 0.f;
  int argmax_idx;
  int last_index = 0;
  int count = 0;
  float max_value = 0.0f;

  for (int n = 0; n < output_shape[1]; n++) {
    argmax_idx = int(
        std::distance(&out_data[n * output_shape[2]],
                      std::max_element(&out_data[n * output_shape[2]],
                                       &out_data[(n + 1) * output_shape[2]])));

    max_value = float(*std::max_element(&out_data[n * output_shape[2]],
                                        &out_data[(n + 1) * output_shape[2]]));

    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      if (argmax_idx > label_list_.size()) {
        FDERROR << "The output index: " << argmax_idx
                << " is larger than the size of label_list: "
                << label_list_.size() << ". Please check the label file!"
                << std::endl;
        return false;
      }
      str_res += label_list_[argmax_idx];
    }
    last_index = argmax_idx;
  }
  score /= (count + 1e-6);
  if (count == 0 || std::isnan(score)) {
    score = 0.f;
  }
  return true;
}

bool RecognizerPostprocessor::Run(const std::vector<FDTensor>& tensors,
                                  std::vector<std::string>* texts,
                                  std::vector<float>* rec_scores) {
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }
  // Recognizer have only 1 output tensor.
  const FDTensor& tensor = tensors[0];
  // For Recognizer, the output tensor shape = [batch, ?, 6625]
  size_t batch = tensor.shape[0];
  size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                             std::multiplies<int>());

  texts->resize(batch);
  rec_scores->resize(batch);
  const float* tensor_data = reinterpret_cast<const float*>(tensor.Data());
  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    if (!SingleBatchPostprocessor(tensor_data, tensor.shape,
                                  &texts->at(i_batch),
                                  &rec_scores->at(i_batch))) {
      return false;
    }
    tensor_data = tensor_data + length;
  }

  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
