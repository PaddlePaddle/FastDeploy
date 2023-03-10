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

#include "paddle2onnx/mapper/nn/rnn.h"
#include <cmath>

namespace paddle2onnx {
REGISTER_MAPPER(rnn, RnnMapper)

int32_t RnnMapper::GetMinOpset(bool verbose) {
  return 7;
}

std::string RnnMapper::ReformWeight(const std::string& weight, const int64_t& size, const std::vector<int64_t>& perm) {
  std::vector<std::string> items;
  for (size_t i = 0; i < perm.size(); i += 2) {
    auto item = helper_->Slice(weight, {1}, {perm[i] * size}, {perm[i + 1] * size});
    items.push_back(item);
  }
  return helper_->Concat(items, 1);
}

std::vector<std::string> RnnMapper::MakeParamInputs(int64_t layer_index) {
  auto weight_list_info = GetInput("WeightList");
  int64_t bidirect_len = is_bidirec_ ? 4 : 2;
  int64_t all_layer_param_len = weight_list_info.size();
  int64_t single_layer_param_len = std::floor(all_layer_param_len / num_layers_);
  int64_t weight_start_idx = layer_index * bidirect_len;
  int64_t weight_end_idx = weight_start_idx + bidirect_len;
  int64_t bias_start_idx = weight_start_idx + std::floor(all_layer_param_len / 2);
  int64_t bias_end_idx = bias_start_idx + bidirect_len;

  std::vector<std::string> unsqueezed_weights;
  for (auto i = weight_start_idx; i < weight_end_idx; ++i) {
    unsqueezed_weights.push_back(helper_->Unsqueeze(weight_list_info[i].name, {0}));
  }
  for (auto i = bias_start_idx; i < bias_end_idx; ++i) {
    unsqueezed_weights.push_back(helper_->Unsqueeze(weight_list_info[i].name, {0}));
  }
  
  std::vector<std::string> input_weight;
  std::vector<std::string> hidden_weight;
  for (size_t i = 0; i < bidirect_len; i += 2) {
    input_weight.push_back(unsqueezed_weights[i]);
  }
  for (size_t i = 1; i < bidirect_len; i += 2) {
    hidden_weight.push_back(unsqueezed_weights[i]);
  }
  std::vector<std::string> input_bias;
  std::vector<std::string> hidden_bias;
  for (size_t i = bidirect_len; i < 2 * bidirect_len; i += 2) {
    input_bias.push_back(unsqueezed_weights[i]);
  }
  for (size_t i = bidirect_len + 1; i < 2 * bidirect_len; i += 2) {
    hidden_bias.push_back(unsqueezed_weights[i]);
  }

  auto input_weight_tensor = helper_->Concat(input_weight, 0);
  auto hidden_weight_tensor = helper_->Concat(hidden_weight, 0);
  auto input_bias_tensor = helper_->Concat(input_bias, 0);
  auto hidden_bias_tensor = helper_->Concat(hidden_bias, 0);

  std::vector<int64_t> reform_permutation;
  if (mode_ == "LSTM") {
    std::vector<int64_t> perm({0, 1, 3, 4, 1, 3});
    reform_permutation.assign(perm.begin(), perm.end());
  } else if (mode_ == "GRU") {
    std::vector<int64_t> perm({1, 2, 0, 1, 2, 3});
    reform_permutation.assign(perm.begin(), perm.end());
  }
  input_weight_tensor = ReformWeight(input_weight_tensor, hidden_size_, reform_permutation);
  hidden_weight_tensor = ReformWeight(hidden_weight_tensor, hidden_size_, reform_permutation);
  input_bias_tensor = ReformWeight(input_bias_tensor, hidden_size_, reform_permutation);
  hidden_bias_tensor = ReformWeight(hidden_bias_tensor, hidden_size_, reform_permutation);
  
  std::vector<std::string> outputs;
  outputs.push_back(input_weight_tensor);
  outputs.push_back(hidden_weight_tensor);
  outputs.push_back(helper_->Concat({input_bias_tensor, hidden_bias_tensor}, 1));
  outputs.push_back("");
  return outputs;
}

std::vector<std::string> RnnMapper::MakeInitParamInputs(int64_t layer_index) {
  std::vector<std::string> outputs;
  auto prestate_info = GetInput("PreState");
  int64_t bidirect_len = is_bidirec_ ? 2 : 1;
  auto init_h = helper_->Slice(prestate_info[0].name, {0}, {layer_index * bidirect_len}, {layer_index * bidirect_len + bidirect_len});
  outputs.push_back(init_h);
  if (mode_ == "GRU") {
    return outputs;
  }
  auto init_c = helper_->Slice(prestate_info[1].name, {0}, {layer_index * bidirect_len}, {layer_index * bidirect_len + bidirect_len});
  outputs.push_back(init_c);
  return outputs;
}

void RnnMapper::Opset7() {
  auto input_info = GetInput("Input");
  auto state_info = GetOutput("State");
  auto out_info = GetOutput("Out");
  auto input = input_info[0].name;
  if (mode_ == "LSTM") {
    std::string h_out = "";
    std::string c_out = "";
    for (auto i = 0; i < num_layers_; ++i) {
      auto param_inputs = MakeParamInputs(i);
      auto init_param_inputs = MakeInitParamInputs(i);
      std::vector<std::string> inputs({input});
      inputs.insert(inputs.end(), param_inputs.begin(), param_inputs.end());
      inputs.insert(inputs.end(), init_param_inputs.begin(), init_param_inputs.end());
      auto node = helper_->MakeNode("LSTM", inputs, 3);
      std::string direction = is_bidirec_ ? "bidirectional" : "forward";
      AddAttribute(node, "direction", direction);
      AddAttribute(node, "hidden_size", hidden_size_);
      input = helper_->Transpose(node->output(0), {0, 2, 1, 3});
      input = helper_->Reshape(input, {0, 0, -1});
      h_out = node->output(1);
      c_out = node->output(2);
    }
    helper_->MakeNode("Identity", {h_out}, {state_info[0].name});
    helper_->MakeNode("Identity", {c_out}, {state_info[1].name});
    helper_->MakeNode("Identity", {input}, {out_info[0].name});
  } else if (mode_ == "GRU") {
    std::string h_out = "";
    for (auto i = 0; i < num_layers_; ++i) {
      auto param_inputs = MakeParamInputs(i);
      auto init_param_inputs = MakeInitParamInputs(i);
      std::vector<std::string> inputs({input});
      inputs.insert(inputs.end(), param_inputs.begin(), param_inputs.end());
      inputs.insert(inputs.end(), init_param_inputs.begin(), init_param_inputs.end());
      auto node = helper_->MakeNode("GRU", inputs, 2);
      std::string direction = is_bidirec_ ? "bidirectional" : "forward";
      AddAttribute(node, "direction", direction);
      AddAttribute(node, "hidden_size", hidden_size_);
      AddAttribute(node, "linear_before_reset", int64_t(1));
      input = helper_->Transpose(node->output(0), {0, 2, 1, 3});
      input = helper_->Reshape(input, {0, 0, -1});
      h_out = node->output(1);
    }
    helper_->MakeNode("Identity", {h_out}, {state_info[0].name});
    helper_->MakeNode("Identity", {input}, {out_info[0].name});
  }
}
}  // namespace paddle2onnx
