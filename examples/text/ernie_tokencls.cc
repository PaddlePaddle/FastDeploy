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
#include <iostream>

#include "compute.h"
#include "fastdeploy/text.h"
#include "tokenizers/ernie_faster_tokenizer.h"

using namespace paddlenlp;

// Only useful for axis = -1
template <typename T>
void Softmax(const fastdeploy::FDTensor& input, fastdeploy::FDTensor* output) {
  auto softmax_func = [](const T* score_vec, T* softmax_vec, int label_num) {
    double score_max = *(std::max_element(score_vec, score_vec + label_num));
    double e_sum = 0;
    for (int j = 0; j < label_num; j++) {
      softmax_vec[j] = std::exp(score_vec[j] - score_max);
      e_sum += softmax_vec[j];
    }
    for (int k = 0; k < label_num; k++) {
      softmax_vec[k] /= e_sum;
    }
  };

  std::vector<int32_t> output_shape;
  for (int i = 0; i < input.shape.size(); ++i) {
    output_shape.push_back(input.shape[i]);
  }
  output->Allocate(output_shape, input.dtype);
  int label_num = output_shape.back();
  int batch_size = input.Numel() / label_num;
  int offset = 0;
  const T* input_ptr = reinterpret_cast<const T*>(input.Data());
  T* output_ptr = reinterpret_cast<T*>(output->Data());
  for (int i = 0; i < batch_size; ++i) {
    softmax_func(input_ptr + offset, output_ptr + offset, label_num);
    offset += label_num;
  }
}

int main() {
  // 1. Define a ernie faster tokenizer
  faster_tokenizer::tokenizers_impl::ErnieFasterTokenizer tokenizer(
      "ernie_vocab.txt");
  std::vector<faster_tokenizer::core::EncodeInput> strings_list = {
      "导航去科技园二号楼", "屏幕亮度为我减小一点吧"};
  std::vector<faster_tokenizer::core::Encoding> encodings;
  tokenizer.EncodeBatchStrings(strings_list, &encodings);
  size_t batch_size = strings_list.size();
  size_t seq_len = encodings[0].GetLen();
  for (auto&& encoding : encodings) {
    std::cout << encoding.DebugString() << std::endl;
  }
  // 2. Initialize runtime
  fastdeploy::RuntimeOption runtime_option;
  runtime_option.SetModelPath("nano_static/model.pdmodel",
                              "nano_static/model.pdiparams");
  fastdeploy::Runtime runtime;
  runtime.Init(runtime_option);

  // 3. Construct input vector
  // 3.1 Convert encodings to input_ids, token_type_ids
  std::vector<int64_t> input_ids, token_type_ids;
  for (int i = 0; i < encodings.size(); ++i) {
    auto&& curr_input_ids = encodings[i].GetIds();
    auto&& curr_type_ids = encodings[i].GetTypeIds();
    input_ids.insert(input_ids.end(), curr_input_ids.begin(),
                     curr_input_ids.end());
    token_type_ids.insert(token_type_ids.end(), curr_type_ids.begin(),
                          curr_type_ids.end());
  }
  // 3.2 Set data to input vector
  std::vector<fastdeploy::FDTensor> inputs(runtime.NumInputs());
  void* inputs_ptrs[] = {input_ids.data(), token_type_ids.data()};
  for (int i = 0; i < runtime.NumInputs(); ++i) {
    inputs[i].SetExternalData({batch_size, seq_len},
                              fastdeploy::FDDataType::INT64, inputs_ptrs[i]);
    inputs[i].name = runtime.GetInputInfo(i).name;
  }

  // 4. Infer
  std::vector<fastdeploy::FDTensor> outputs(runtime.NumOutputs());
  runtime.Infer(inputs, &outputs);

  // 5. Postprocess
  fastdeploy::FDTensor domain_probs, intent_probs;
  Softmax<float>(outputs[0], &domain_probs);
  Softmax<float>(outputs[1], &intent_probs);

  Eigen::DefaultDevice dev;
  fastdeploy::FDTensor domain_max_probs, intent_max_probs;
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::MaxFunctor>(
      dev, domain_probs, &domain_max_probs, {1});
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::MaxFunctor>(
      dev, intent_probs, &intent_max_probs, {1});

  // 6. Print result
  domain_max_probs.PrintInfo();
  intent_max_probs.PrintInfo();
  return 0;
}