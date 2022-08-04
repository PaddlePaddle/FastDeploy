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
  std::cout << "outputs size: " << outputs.size() << std::endl;
  for (auto&& output : outputs) {
    std::cout << "shape: (";
    for (auto&& s : output.shape) {
      std::cout << s << ", ";
    }
    std::cout << ")" << std::endl;
  }

  // 6. Print result

  return 0;
}