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

#include "uie.h"
#include <algorithm>
#include <queue>

#include "utils/utf8.h"  // faster_tokenizer helper funciton

static std::string DBC2SBC(const std::string& content) {
  std::string result;
  size_t content_utf8_len = 0;
  while (content_utf8_len < content.length()) {
    uint32_t content_char;
    auto content_char_width = faster_tokenizer::utils::UTF8ToUInt32(
        content.data() + content_utf8_len, &content_char);
    content_char = faster_tokenizer::utils::UTF8ToUnicode(content_char);
    if (content_char == 0x3000) {
      content_char = 0x0020;
    } else {
      content_char -= 0xfee0;
    }
    if (content_char >= 0x0021 && content_char <= 0x7e) {
      result.append(content.data() + content_utf8_len, content_char_width);
    } else {
      char dst_char[5] = {0};
      uint32_t utf8_uint32 =
          faster_tokenizer::utils::UnicodeToUTF8(content_char);
      uint32_t utf8_char_count =
          faster_tokenizer::utils::UnicodeToUTF8Char(utf8_uint32, dst_char);
      dst_char[utf8_char_count] = '\0';
      result.append(dst_char);
    }
    content_utf8_len += content_char_width;
  }
  return result;
}

void Schema::CreateRoot(const std::string& name) {
  root_ = fastdeploy::utils::make_unique<SchemaNode>(name);
}

Schema::Schema(const std::string& schema, const std::string& name) {
  CreateRoot(name);
  root_->AddChild(schema);
}

Schema::Schema(const std::vector<std::string>& schema_list,
               const std::string& name) {
  CreateRoot(name);
  for (const auto& schema : schema_list) {
    root_->AddChild(schema);
  }
}

Schema::Schema(
    const std::unordered_map<std::string, std::vector<std::string>>& schema_map,
    const std::string& name) {
  CreateRoot(name);
  for (auto& schema_item : schema_map) {
    root_->AddChild(schema_item.first, schema_item.second);
  }
}

UIEModel::UIEModel(const std::string& model_file,
                   const std::string& params_file,
                   const std::string& vocab_file, double position_prob,
                   size_t max_length, const std::vector<std::string>& schema)
    : max_length_(max_length),
      position_prob_(position_prob),
      tokenizer_(vocab_file) {
  runtime_option_.SetModelPath(model_file, params_file);
  runtime_.Init(runtime_option_);
  SetSchema(schema);
  tokenizer_.EnableTruncMethod(
      max_length, 0, faster_tokenizer::core::Direction::RIGHT,
      faster_tokenizer::core::TruncStrategy::LONGEST_FIRST);
}

UIEModel::UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, double position_prob, size_t max_length,
    const std::unordered_map<std::string, std::vector<std::string>>& schema)
    : max_length_(max_length),
      position_prob_(position_prob),
      tokenizer_(vocab_file) {
  runtime_option_.SetModelPath(model_file, params_file);
  runtime_.Init(runtime_option_);
  SetSchema(schema);
  tokenizer_.EnableTruncMethod(
      max_length, 0, faster_tokenizer::core::Direction::RIGHT,
      faster_tokenizer::core::TruncStrategy::LONGEST_FIRST);
}

void UIEModel::SetSchema(const std::vector<std::string>& schema) {
  schema_ = fastdeploy::utils::make_unique<Schema>(schema);
}

void UIEModel::SetSchema(
    const std::unordered_map<std::string, std::vector<std::string>>& schema) {
  schema_ = fastdeploy::utils::make_unique<Schema>(schema);
}

void UIEModel::AutoSplitter(
    const std::vector<std::string>& texts, size_t max_length,
    std::vector<std::string>* short_texts,
    std::unordered_map<size_t, std::vector<size_t>>* input_mapping) {
  size_t cnt_org = 0;
  size_t cnt_short = 0;
  for (auto& text : texts) {
    auto text_len = text.length();
    if (text_len <= max_length) {
      short_texts->push_back(text);
      if (input_mapping->count(cnt_org) == 0) {
        (*input_mapping)[cnt_org] = {cnt_short};
      } else {
        (*input_mapping)[cnt_org].push_back(cnt_short);
      }
      cnt_short += 1;
    } else {
      for (size_t start = 0; start < text_len; start += max_length) {
        size_t end = start + max_length;
        if (end > text_len) {
          end = text_len;
        }
        short_texts->emplace_back(text.data() + start, end - start);
      }
      auto short_idx = cnt_short;
      cnt_short += text_len / max_length;
      if (text_len % max_length != 0) {
        ++cnt_short;
      }
      std::vector<size_t> temp_text_id(cnt_short - short_idx);
      std::iota(temp_text_id.begin(), temp_text_id.end(), short_idx);
      if (input_mapping->count(cnt_org) == 0) {
        (*input_mapping)[cnt_org] = std::move(temp_text_id);
      } else {
        (*input_mapping)[cnt_org].insert((*input_mapping)[cnt_org].end(),
                                         temp_text_id.begin(),
                                         temp_text_id.end());
      }
    }
    cnt_org += 1;
  }
}

void UIEModel::PredictUIEInput(const std::vector<std::string>& input_texts,
                               const std::vector<std::string>& prompts) {
  // 1. Shortten the input texts and prompts
  auto max_predict_len =
      max_length_ - 3 -
      std::max_element(prompts.begin(), prompts.end(),
                       [](const std::string& lhs, const std::string& rhs) {
                         return lhs.length() < rhs.length();
                       })
          ->length();
  std::vector<std::string> short_texts;
  std::unordered_map<size_t, std::vector<size_t>> input_mapping;
  AutoSplitter(input_texts, max_predict_len, &short_texts, &input_mapping);

  std::vector<std::string> short_texts_prompts;
  for (auto& item : input_mapping) {
    short_texts_prompts.insert(short_texts_prompts.end(), item.second.size(),
                               prompts[item.first]);
  }
  std::vector<faster_tokenizer::core::EncodeInput> text_pair_input;
  for (int i = 0; i < short_texts.size(); ++i) {
    text_pair_input.emplace_back(std::pair<std::string, std::string>(
        short_texts[i], short_texts_prompts[i]));
  }

  // 2. Tokenize the short texts and short prompts
  std::vector<faster_tokenizer::core::Encoding> encodings;
  tokenizer_.EncodeBatchStrings(text_pair_input, &encodings);

  // 3. Construct the input vector tensor
  // 3.1 Convert encodings to input_ids, token_type_ids, position_ids, attn_mask
  std::vector<int64_t> input_ids, token_type_ids, position_ids, attn_mask;
  for (int i = 0; i < encodings.size(); ++i) {
    auto&& curr_input_ids = encodings[i].GetIds();
    auto&& curr_type_ids = encodings[i].GetTypeIds();
    auto&& curr_attn_mask = encodings[i].GetAttentionMask();
    input_ids.insert(input_ids.end(), curr_input_ids.begin(),
                     curr_input_ids.end());
    token_type_ids.insert(token_type_ids.end(), curr_type_ids.begin(),
                          curr_type_ids.end());
    attn_mask.insert(attn_mask.end(), curr_attn_mask.begin(),
                     curr_attn_mask.end());
    std::vector<int64_t> curr_position_ids(curr_input_ids.size());
    std::iota(curr_position_ids.begin(), curr_position_ids.end(), 0);
    position_ids.insert(position_ids.end(), curr_position_ids.begin(),
                        curr_position_ids.end());
  }
  // 3.2 Set data to input vector
  int64_t batch_size = input_texts.size();
  int64_t seq_len = input_ids.size() / batch_size;
  std::vector<fastdeploy::FDTensor> inputs(runtime_.NumInputs());
  int64_t* inputs_ptrs[] = {input_ids.data(), token_type_ids.data(),
                            position_ids.data(), attn_mask.data()};
  for (int i = 0; i < runtime_.NumInputs(); ++i) {
    inputs[i].SetExternalData({batch_size, seq_len},
                              fastdeploy::FDDataType::INT64, inputs_ptrs[i]);
    inputs[i].name = runtime_.GetInputInfo(i).name;
  }

  std::vector<float> start_probs, end_probs;
  std::vector<fastdeploy::FDTensor> outputs(runtime_.NumOutputs());
  // 4. Infer
  runtime_.Infer(inputs, &outputs);
  auto* start_prob = reinterpret_cast<float*>(outputs[0].Data());
  auto* end_prob = reinterpret_cast<float*>(outputs[1].Data());
  start_probs.insert(start_probs.end(), start_prob,
                     start_prob + outputs[0].Numel());
  end_probs.insert(end_probs.end(), end_prob, end_prob + outputs[0].Numel());

  // 5. Postprocess
}

void UIEModel::Predict(const std::vector<std::string>& texts,
                       std::vector<UIEResult>* results) {
  std::queue<SchemaNode> nodes;
  for (auto& node : schema_->root_->children_) {
    nodes.push(node);
  }
  while (!nodes.empty()) {
    auto& node = nodes.front();
    nodes.pop();
    std::vector<std::vector<size_t>> input_mapping;
    size_t idx = 0;
    std::vector<std::string> input_texts;
    std::vector<std::string> prompts;
    // 1. Construct input data from raw text
    if (node.prefix_.empty()) {
      for (int i = 0; i < texts.size(); ++i) {
        input_texts.push_back(texts[i]);
        prompts.push_back(DBC2SBC(node.name_));
        input_mapping.push_back({idx});
        idx += 1;
      }
    } else {
      for (int i = 0; i < texts.size(); ++i) {
        if (node.prefix_[i].size() == 0) {
          input_mapping.push_back({});
        } else {
          for (auto&& pre : node.prefix_[i]) {
            input_texts.push_back(texts[i]);
            prompts.push_back(DBC2SBC(pre + node.name_));
          }
          input_mapping.push_back({});
          input_mapping.back().resize(node.prefix_[i].size());
          std::iota(input_mapping.back().begin(), input_mapping.back().end(),
                    idx);
          idx += node.prefix_[i].size();
        }
      }
    }

    // 2. Predict from UIEInput
    PredictUIEInput(input_texts, prompts);

    // 3. Postprocess
    for (auto& node_child : node.children_) {
      nodes.push(node_child);
    }
  }
}
