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
  std::vector<fastdeploy::FDTensor> inputs(runtime.NumInputs());
  for (int i = 0; i < runtime.NumInputs(); ++i) {
    inputs[i].dtype = fastdeploy::FDDataType::INT64;
    inputs[i].shape = {batch_size, seq_len};
    inputs[i].name = runtime.GetInputInfo(i).name;
    inputs[i].data.resize(sizeof(int64_t) * batch_size * seq_len);
  }

  // Convert encodings to input_ids, token_type_ids
  std::vector<int64_t> input_ids, token_type_ids;
  for (int i = 0; i < encodings.size(); ++i) {
    auto&& curr_input_ids = encodings[i].GetIds();
    auto&& curr_type_ids = encodings[i].GetTypeIds();
    input_ids.insert(input_ids.end(), curr_input_ids.begin(),
                     curr_input_ids.end());
    token_type_ids.insert(token_type_ids.end(), curr_type_ids.begin(),
                          curr_type_ids.end());
  }

  memcpy(inputs[0].data.data(), input_ids.data(), inputs[0].data.size());
  memcpy(inputs[1].data.data(), token_type_ids.data(), inputs[1].data.size());

  // 4. Infer
  std::vector<fastdeploy::FDTensor> outputs(runtime.NumOutputs());
  runtime.Infer(inputs, &outputs);

  // 5. Postprocess
  // domain_max_value = np.max(domain_logits, axis=1, keepdims=True)
  // intent_max_value = np.max(intent_logits, axis=1, keepdims=True)
  fastdeploy::FDTensor domain_max_value, intent_max_value;
  Eigen::DefaultDevice dev;
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::MaxFunctor>(
      dev, outputs[0], &domain_max_value, {1});
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::MaxFunctor>(
      dev, outputs[1], &intent_max_value, {1});
  // domain_exp_data = np.exp(domain_logits - domain_max_value)
  // intent_exp_data = np.exp(intent_logits - intent_max_value)
  fastdeploy::FDTensor domain_exp_data, intent_exp_data;
  // Broadcast and diff
  fastdeploy::CommonElementwiseBroadcastForward<fastdeploy::SubFunctor<float>,
                                                float>(
      outputs[0], domain_max_value, &domain_exp_data,
      fastdeploy::SubFunctor<float>(), 0);
  fastdeploy::CommonElementwiseBroadcastForward<fastdeploy::SubFunctor<float>,
                                                float>(
      outputs[1], intent_max_value, &intent_exp_data,
      fastdeploy::SubFunctor<float>(), 0);
  // domain_exp_data = np.exp(domain_logits - domain_max_value)
  // intent_exp_data = np.exp(intent_logits - intent_max_value)
  float* domain_exp_data_ptr = reinterpret_cast<float*>(domain_exp_data.Data());
  float* intent_exp_data_ptr = reinterpret_cast<float*>(intent_exp_data.Data());
  auto trans = [](float a) { return std::exp(a); };
  std::transform(domain_exp_data_ptr,
                 domain_exp_data_ptr + domain_exp_data.Numel(),
                 domain_exp_data_ptr, trans);
  std::transform(intent_exp_data_ptr,
                 intent_exp_data_ptr + intent_exp_data.Numel(),
                 intent_exp_data_ptr, trans);
  // domain_probs = domain_exp_data / np.sum(domain_exp_data, axis=1,
  // keepdims=True)
  // intent_probs = intent_exp_data / np.sum(intent_exp_data, axis=1,
  // keepdims=True)
  fastdeploy::FDTensor domain_exp_data_sum, intent_exp_data_sum;
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::SumFunctor>(
      dev, domain_exp_data, &domain_exp_data_sum, {1});
  fastdeploy::ReduceFunctor<float, 2, 1, fastdeploy::SumFunctor>(
      dev, intent_exp_data, &intent_exp_data_sum, {1});

  fastdeploy::FDTensor domain_probs, intent_probs;
  fastdeploy::CommonElementwiseBroadcastForward<fastdeploy::DivFunctor<float>,
                                                float>(
      domain_exp_data, domain_exp_data_sum, &domain_probs,
      fastdeploy::DivFunctor<float>(), 0);
  fastdeploy::CommonElementwiseBroadcastForward<fastdeploy::DivFunctor<float>,
                                                float>(
      intent_exp_data, intent_exp_data_sum, &intent_probs,
      fastdeploy::DivFunctor<float>(), 0);

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