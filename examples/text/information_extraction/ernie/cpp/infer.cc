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
#include <sstream>

#include "fastdeploy/function/reduce.h"
#include "fastdeploy/function/softmax.h"
#include "fastdeploy/text.h"
#include "tokenizers/ernie_faster_tokenizer.h"

using namespace paddlenlp;

void LoadTransitionFromFile(const std::string& file,
                            std::vector<float>* transitions, int* num_tags) {
  std::ifstream fin(file);
  std::string curr_transition;
  float transition;
  int i = 0;
  while (fin) {
    std::getline(fin, curr_transition);
    std::istringstream iss(curr_transition);
    while (iss) {
      iss >> transition;
      transitions->push_back(transition);
    }
    if (curr_transition != "") {
      ++i;
    }
  }
  *num_tags = i;
}

template <typename T>
void ViterbiDecode(const fastdeploy::FDTensor& slot_logits,
                   const fastdeploy::FDTensor& trans,
                   fastdeploy::FDTensor* best_path) {
  int batch_size = slot_logits.shape[0];
  int seq_len = slot_logits.shape[1];
  int num_tags = slot_logits.shape[2];
  best_path->Allocate({batch_size, seq_len}, fastdeploy::FDDataType::INT64);

  const T* slot_logits_ptr = reinterpret_cast<const T*>(slot_logits.Data());
  const T* trans_ptr = reinterpret_cast<const T*>(trans.Data());
  int64_t* best_path_ptr = reinterpret_cast<int64_t*>(best_path->Data());
  std::vector<T> scores(num_tags);
  std::copy(slot_logits_ptr, slot_logits_ptr + num_tags, scores.begin());
  std::vector<std::vector<T>> M(num_tags, std::vector<T>(num_tags));
  for (int b = 0; b < batch_size; ++b) {
    std::vector<std::vector<int>> paths;
    const T* curr_slot_logits_ptr = slot_logits_ptr + b * seq_len * num_tags;
    int64_t* curr_best_path_ptr = best_path_ptr + b * seq_len;
    for (int t = 1; t < seq_len; t++) {
      for (size_t i = 0; i < num_tags; i++) {
        for (size_t j = 0; j < num_tags; j++) {
          auto trans_idx = i * num_tags * num_tags + j * num_tags;
          auto slot_logit_idx = t * num_tags + j;
          M[i][j] = scores[i] + trans_ptr[trans_idx] +
                    curr_slot_logits_ptr[slot_logit_idx];
        }
      }
      std::vector<int> idxs;
      for (size_t i = 0; i < num_tags; i++) {
        T max = 0.0f;
        int idx = 0;
        for (size_t j = 0; j < num_tags; j++) {
          if (M[j][i] > max) {
            max = M[j][i];
            idx = j;
          }
        }
        scores[i] = max;
        idxs.push_back(idx);
      }
      paths.push_back(idxs);
    }
    int scores_max_index = 0;
    float scores_max = 0.0f;
    for (size_t i = 0; i < scores.size(); i++) {
      if (scores[i] > scores_max) {
        scores_max = scores[i];
        scores_max_index = i;
      }
    }
    curr_best_path_ptr[seq_len - 1] = scores_max_index;
    for (int i = seq_len - 2; i >= 0; i--) {
      int index = curr_best_path_ptr[i + 1];
      curr_best_path_ptr[i] = paths[i][index];
    }
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
  fastdeploy::Softmax(outputs[0], &domain_probs);
  fastdeploy::Softmax(outputs[1], &intent_probs);

  fastdeploy::FDTensor domain_max_probs, intent_max_probs;
  fastdeploy::Max(domain_probs, &domain_max_probs, {-1}, true);
  fastdeploy::Max(intent_probs, &intent_max_probs, {-1}, true);

  std::vector<float> transition;
  int num_tags;
  LoadTransitionFromFile("joint_transition.txt", &transition, &num_tags);
  fastdeploy::FDTensor trans;
  trans.SetExternalData({num_tags, num_tags}, fastdeploy::FDDataType::FP32,
                        transition.data());

  fastdeploy::FDTensor best_path;
  ViterbiDecode<float>(outputs[2], trans, &best_path);
  // 6. Print result
  domain_max_probs.PrintInfo();
  intent_max_probs.PrintInfo();

  batch_size = best_path.shape[0];
  seq_len = best_path.shape[1];
  const int64_t* best_path_ptr =
      reinterpret_cast<const int64_t*>(best_path.Data());
  for (int i = 0; i < batch_size; ++i) {
    std::cout << "best_path[" << i << "] = ";
    for (int j = 0; j < seq_len; ++j) {
      std::cout << best_path_ptr[i * seq_len + j] << ", ";
    }
    std::cout << std::endl;
  }
  best_path.PrintInfo();
  return 0;
}
