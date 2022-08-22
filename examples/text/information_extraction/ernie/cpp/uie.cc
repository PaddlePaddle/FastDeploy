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
#include <codecvt>
#include <locale>
#include <queue>
#include <sstream>

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
    if (!(content_char >= 0x0021 && content_char <= 0x7e)) {
      result.append(content.data() + content_utf8_len, content_char_width);
    } else {
      char dst_char[5] = {0};
      uint32_t utf8_uint32 =
          faster_tokenizer::utils::UnicodeToUTF8(content_char);
      uint32_t utf8_char_count =
          faster_tokenizer::utils::UnicodeToUTF8Char(utf8_uint32, dst_char);
      result.append(dst_char, utf8_char_count);
    }
    content_utf8_len += content_char_width;
  }
  return result;
}

// Will remove to faster_tokenizer utils
static void CharToByteOffsetMap(const std::string& seq,
                                std::vector<uint32_t>* offset_mapping) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32seq = conv.from_bytes(seq);
  uint32_t index = 0;
  offset_mapping->reserve(u32seq.length() * 4);
  for (int i = 0; i < u32seq.length(); ++i) {
    offset_mapping->push_back(index);
    auto utf8_len = faster_tokenizer::utils::GetUTF8CharLen(u32seq[i]);
    index += utf8_len;
  }
  offset_mapping->push_back(index);
}

static std::ostream& PrintResult(std::ostream& os, const UIEResult& result,
                                 int tab_size) {
  constexpr int TAB_OFFSET = 4;
  // Print text
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "text: " << result.text_ << "\n";

  // Print probability
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "probability: " << result.probability_ << "\n";

  // Print start
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "start: " << result.start_ << "\n";

  // Print end
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "end: " << result.end_ << "\n";

  // Print relation
  if (result.relation_.size() > 0) {
    for (int i = 0; i < tab_size; ++i) {
      os << " ";
    }
    os << "relation:\n";
    for (auto&& curr_relation : result.relation_) {
      for (int i = 0; i < tab_size + TAB_OFFSET; ++i) {
        os << " ";
      }
      os << curr_relation.first << ":\n";
      for (int i = 0; i < curr_relation.second.size(); ++i) {
        PrintResult(os, curr_relation.second[i],
                    tab_size + TAB_OFFSET + TAB_OFFSET);
      }
    }
  }
  os << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const UIEResult& result) {
  return PrintResult(os, result, 0);
}

std::ostream& operator<<(
    std::ostream& os,
    const std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>&
        results) {
  os << "The result:\n";
  for (int i = 0; i < results.size(); ++i) {
    for (auto&& curr_result : results[i]) {
      os << curr_result.first << ": \n";
      for (auto&& uie_result : curr_result.second) {
        PrintResult(os, uie_result, 4);
      }
    }
    os << std::endl;
  }
  return os;
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
    const std::unordered_map<std::string, std::vector<SchemaNode>>& schema_map,
    const std::string& name) {
  CreateRoot(name);
  for (auto& schema_item : schema_map) {
    root_->AddChild(schema_item.first, schema_item.second);
  }
}

UIEModel::UIEModel(const std::string& model_file,
                   const std::string& params_file,
                   const std::string& vocab_file, float position_prob,
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
    const std::string& vocab_file, float position_prob, size_t max_length,
    const std::unordered_map<std::string, std::vector<SchemaNode>>& schema)
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
    const std::unordered_map<std::string, std::vector<SchemaNode>>& schema) {
  schema_ = fastdeploy::utils::make_unique<Schema>(schema);
}

void UIEModel::AutoSplitter(
    const std::vector<std::string>& texts, size_t max_length,
    std::vector<std::string>* short_texts,
    std::unordered_map<size_t, std::vector<size_t>>* input_mapping) {
  size_t cnt_org = 0;
  size_t cnt_short = 0;
  for (auto& text : texts) {
    auto text_len = faster_tokenizer::utils::GetUnicodeLenFromUTF8(
        text.c_str(), text.length());
    if (text_len <= max_length) {
      short_texts->push_back(text);
      if (input_mapping->count(cnt_org) == 0) {
        (*input_mapping)[cnt_org] = {cnt_short};
      } else {
        (*input_mapping)[cnt_org].push_back(cnt_short);
      }
      cnt_short += 1;
    } else {
      std::vector<uint32_t> offset_mapping;
      CharToByteOffsetMap(text, &offset_mapping);
      for (size_t start = 0; start < text_len; start += max_length) {
        size_t end = start + max_length;
        if (end > text_len) {
          end = text_len;
        }
        auto unicode_start = offset_mapping[start];
        auto unicode_end = offset_mapping[end];
        short_texts->emplace_back(text.data() + unicode_start,
                                  unicode_end - unicode_start);
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

void UIEModel::GetCandidateIdx(
    const float* probs, int64_t batch_size, int64_t seq_len,
    std::vector<std::vector<std::pair<int64_t, float>>>* candidate_idx_prob,
    float threshold) const {
  for (int i = 0; i < batch_size; ++i) {
    candidate_idx_prob->push_back({});
    for (int j = 0; j < seq_len; ++j) {
      if (probs[i * seq_len + j] > threshold) {
        candidate_idx_prob->back().push_back({j, probs[i * seq_len + j]});
      }
    }
  }
}

bool UIEModel::IdxProbCmp::operator()(
    const std::pair<IDX_PROB, IDX_PROB>& lhs,
    const std::pair<IDX_PROB, IDX_PROB>& rhs) const {
  if (lhs.first.first == rhs.first.first) {
    return lhs.second.first < rhs.second.first;
  }
  return lhs.first.first < rhs.first.first;
}

void UIEModel::GetSpan(const std::vector<IDX_PROB>& start_idx_prob,
                       const std::vector<IDX_PROB>& end_idx_prob,
                       SPAN_SET* span_set) const {
  size_t start_pointer = 0;
  size_t end_pointer = 0;
  size_t len_start = start_idx_prob.size();
  size_t len_end = end_idx_prob.size();
  while (start_pointer < len_start && end_pointer < len_end) {
    if (start_idx_prob[start_pointer].first ==
        end_idx_prob[end_pointer].first) {
      span_set->insert(std::make_pair(start_idx_prob[start_pointer],
                                      end_idx_prob[end_pointer]));
      ++start_pointer;
      ++end_pointer;
    } else if (start_idx_prob[start_pointer].first <
               end_idx_prob[end_pointer].first) {
      span_set->insert(std::make_pair(start_idx_prob[start_pointer],
                                      end_idx_prob[end_pointer]));
      ++start_pointer;
    } else {
      ++end_pointer;
    }
  }
}
void UIEModel::GetSpanIdxAndProbs(
    const SPAN_SET& span_set,
    const std::vector<faster_tokenizer::core::Offset>& offset_mapping,
    std::vector<SpanIdx>* span_idxs, std::vector<float>* probs) const {
  auto first_sep_idx =
      std::find_if(offset_mapping.begin() + 1, offset_mapping.end(),
                   [](const faster_tokenizer::core::Offset& offset) {
                     return offset == faster_tokenizer::core::Offset(0, 0);
                   });
  auto prompt_end_token_id =
      std::distance(offset_mapping.begin(), first_sep_idx) - 1;
  for (auto&& span_item : span_set) {
    probs->push_back(span_item.first.second * span_item.second.second);
    auto start_id = offset_mapping[span_item.first.first].first;
    auto end_id = offset_mapping[span_item.second.first].second;
    bool is_prompt = span_item.second.first <= prompt_end_token_id &&
                     span_item.second.first > 0;
    span_idxs->push_back({{start_id, end_id}, is_prompt});
  }
}

void UIEModel::ConvertSpanToUIEResult(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& prompts,
    const std::vector<std::vector<SpanIdx>>& span_idxs,
    const std::vector<std::vector<float>>& probs,
    std::vector<std::vector<UIEResult>>* results) const {
  auto batch_size = texts.size();
  for (int i = 0; i < batch_size; ++i) {
    std::vector<UIEResult> result_list;
    if (span_idxs[i].size() == 0) {
      results->push_back({});
      continue;
    }
    auto&& text = texts[i];
    auto&& prompt = prompts[i];
    for (int j = 0; j < span_idxs[i].size(); ++j) {
      auto start = span_idxs[i][j].offset_.first;
      auto end = span_idxs[i][j].offset_.second;
      std::string span_text;
      std::vector<uint32_t> offset_mapping;
      if (span_idxs[i][j].is_prompt_) {
        CharToByteOffsetMap(prompt, &offset_mapping);
        auto byte_start = offset_mapping[start];
        auto byte_end = offset_mapping[end];
        span_text = prompt.substr(byte_start, byte_end - byte_start);
        // Indicate cls task
        start = 0;
        end = 0;
      } else {
        CharToByteOffsetMap(text, &offset_mapping);
        auto byte_start = offset_mapping[start];
        auto byte_end = offset_mapping[end];
        span_text = text.substr(byte_start, byte_end - byte_start);
      }
      result_list.emplace_back(start, end, probs[i][j], span_text);
    }
    results->push_back(result_list);
  }
}

void UIEModel::AutoJoiner(
    const std::vector<std::string>& short_texts,
    const std::unordered_map<size_t, std::vector<size_t>>& input_mapping,
    std::vector<std::vector<UIEResult>>* results) {
  bool is_cls_task = false;
  // 1. Detect if it's a cls task
  for (auto&& short_result : *results) {
    if (short_result.size() == 0) {
      continue;
    } else if (short_result[0].start_ == 0 && short_result[0].end_) {
      is_cls_task = true;
      break;
    } else {
      break;
    }
  }
  // 2. Get the final result
  std::vector<std::vector<UIEResult>> final_result;
  if (is_cls_task) {
    for (auto&& input_mapping_item : input_mapping) {
      auto curr_mapping = input_mapping_item.second;
      std::unordered_map<std::string, std::pair<int, float>> cls_options;
      for (auto&& result_idx : curr_mapping) {
        if ((*results)[result_idx].size() == 0) {
          continue;
        }
        auto&& text = (*results)[result_idx].front().text_;
        auto&& probability = (*results)[result_idx].front().probability_;
        if (cls_options.count(text) == 0) {
          cls_options[text] = std::make_pair(1, probability);
        } else {
          cls_options[text].first += 1;
          cls_options[text].second += probability;
        }
      }
      std::vector<UIEResult> result_list;
      if (cls_options.size() > 0) {
        auto max_iter = std::max_element(
            cls_options.begin(), cls_options.end(),
            [](const std::pair<std::string, std::pair<int, float>>& lhs,
               const std::pair<std::string, std::pair<int, float>>& rhs) {
              return lhs.second.second < rhs.second.second;
            });
        result_list.emplace_back(
            0, 0, max_iter->second.second / max_iter->second.first,
            max_iter->first);
      }
      final_result.push_back(result_list);
    }
  } else {
    for (auto&& input_mapping_item : input_mapping) {
      auto curr_mapping = input_mapping_item.second;
      size_t offset = 0;
      std::vector<UIEResult> result_list;
      for (auto&& result_idx : curr_mapping) {
        if (result_idx == 0) {
          result_list = std::move((*results)[result_idx]);
          offset += faster_tokenizer::utils::GetUnicodeLenFromUTF8(
              short_texts[result_idx].c_str(), short_texts[result_idx].size());
        } else {
          for (auto&& curr_result : (*results)[result_idx]) {
            curr_result.start_ += offset;
            curr_result.end_ += offset;
          }
          offset += faster_tokenizer::utils::GetUnicodeLenFromUTF8(
              short_texts[result_idx].c_str(), short_texts[result_idx].size());
          result_list.insert(result_list.end(), (*results)[result_idx].begin(),
                             (*results)[result_idx].end());
        }
      }
      final_result.push_back(result_list);
    }
  }
  *results = std::move(final_result);
}

void UIEModel::PredictUIEInput(const std::vector<std::string>& input_texts,
                               const std::vector<std::string>& prompts,
                               std::vector<std::vector<UIEResult>>* results) {
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
        short_texts_prompts[i], short_texts[i]));
  }

  // 2. Tokenize the short texts and short prompts
  std::vector<faster_tokenizer::core::Encoding> encodings;
  tokenizer_.EncodeBatchStrings(text_pair_input, &encodings);
  // 3. Construct the input vector tensor
  // 3.1 Convert encodings to input_ids, token_type_ids, position_ids, attn_mask
  std::vector<int64_t> input_ids, token_type_ids, position_ids, attn_mask;
  std::vector<std::vector<faster_tokenizer::core::Offset>> offset_mapping;
  for (int i = 0; i < encodings.size(); ++i) {
    auto&& curr_input_ids = encodings[i].GetIds();
    auto&& curr_type_ids = encodings[i].GetTypeIds();
    auto&& curr_attn_mask = encodings[i].GetAttentionMask();
    auto&& curr_offsets = encodings[i].GetOffsets();
    input_ids.insert(input_ids.end(), curr_input_ids.begin(),
                     curr_input_ids.end());
    token_type_ids.insert(token_type_ids.end(), curr_type_ids.begin(),
                          curr_type_ids.end());
    attn_mask.insert(attn_mask.end(), curr_attn_mask.begin(),
                     curr_attn_mask.end());
    offset_mapping.push_back(curr_offsets);
    std::vector<int64_t> curr_position_ids(curr_input_ids.size());
    std::iota(curr_position_ids.begin(), curr_position_ids.end(), 0);
    position_ids.insert(position_ids.end(), curr_position_ids.begin(),
                        curr_position_ids.end());
  }

  // 3.2 Set data to input vector
  int64_t batch_size = short_texts.size();
  int64_t seq_len = input_ids.size() / batch_size;
  std::vector<fastdeploy::FDTensor> inputs(runtime_.NumInputs());
  int64_t* inputs_ptrs[] = {input_ids.data(), token_type_ids.data(),
                            position_ids.data(), attn_mask.data()};
  for (int i = 0; i < runtime_.NumInputs(); ++i) {
    inputs[i].SetExternalData({batch_size, seq_len},
                              fastdeploy::FDDataType::INT64, inputs_ptrs[i]);
    inputs[i].name = runtime_.GetInputInfo(i).name;
  }

  std::vector<fastdeploy::FDTensor> outputs(runtime_.NumOutputs());
  // 4. Infer
  runtime_.Infer(inputs, &outputs);
  auto* start_prob = reinterpret_cast<float*>(outputs[0].Data());
  auto* end_prob = reinterpret_cast<float*>(outputs[1].Data());

  // 5. Postprocess
  std::vector<std::vector<std::pair<int64_t, float>>> start_candidate_idx_prob,
      end_candidate_idx_prob;

  GetCandidateIdx(start_prob, outputs[0].shape[0], outputs[0].shape[1],
                  &start_candidate_idx_prob, position_prob_);
  GetCandidateIdx(end_prob, outputs[1].shape[0], outputs[1].shape[1],
                  &end_candidate_idx_prob, position_prob_);
  SPAN_SET span_set;
  std::vector<std::vector<float>> probs(batch_size);
  std::vector<std::vector<SpanIdx>> span_idxs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    GetSpan(start_candidate_idx_prob[i], end_candidate_idx_prob[i], &span_set);
    GetSpanIdxAndProbs(span_set, offset_mapping[i], &span_idxs[i], &probs[i]);
    span_set.clear();
  }
  ConvertSpanToUIEResult(short_texts, short_texts_prompts, span_idxs, probs,
                         results);
  AutoJoiner(short_texts, input_mapping, results);
}

void UIEModel::Predict(
    const std::vector<std::string>& texts,
    std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>*
        results) {
  std::queue<SchemaNode> nodes;
  for (auto& node : schema_->root_->children_) {
    nodes.push(node);
  }
  results->resize(texts.size());
  while (!nodes.empty()) {
    auto node = nodes.front();
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
          auto prefix_len = node.prefix_[i].size();
          input_mapping.push_back({});
          input_mapping.back().resize(prefix_len);
          std::iota(input_mapping.back().begin(), input_mapping.back().end(),
                    idx);
          idx += prefix_len;
        }
      }
    }

    // 2. Predict from UIEInput
    std::vector<std::vector<UIEResult>> results_list;
    PredictUIEInput(input_texts, prompts, &results_list);
    // 3. Postprocess
    std::vector<std::vector<UIEResult*>> relations;
    relations.resize(texts.size());
    if (node.relations_.size() == 0) {
      for (int i = 0; i < input_mapping.size(); ++i) {
        auto&& input_mapping_item = input_mapping[i];
        auto& curr_result = (*results)[i];
        for (auto&& idx : input_mapping_item) {
          if (results_list[idx].size() == 0) {
            continue;
          }
          if (curr_result.count(node.name_) == 0) {
            curr_result[node.name_] = results_list[idx];
          } else {
            curr_result[node.name_].insert(curr_result[node.name_].end(),
                                           results_list[idx].begin(),
                                           results_list[idx].end());
          }
        }
        if (curr_result.count(node.name_) > 0) {
          for (auto&& curr_result_ref : curr_result[node.name_]) {
            relations[i].push_back(&curr_result_ref);
          }
        }
      }
    } else {
      auto& new_relations = node.relations_;
      for (int i = 0; i < input_mapping.size(); ++i) {
        auto&& input_mapping_item = input_mapping[i];
        for (int j = 0; j < input_mapping_item.size(); ++j) {
          auto idx = input_mapping_item[j];
          if (results_list[idx].size() == 0) {
            continue;
          }
          if (new_relations[i][j]->relation_.count(node.name_) == 0) {
            new_relations[i][j]->relation_[node.name_] = results_list[idx];
          } else {
            auto& curr_result = new_relations[i][j]->relation_[node.name_];
            curr_result.insert(curr_result.end(), results_list[idx].begin(),
                               results_list[idx].end());
          }
        }
      }
      for (int i = 0; i < new_relations.size(); ++i) {
        for (int j = 0; j < new_relations[i].size(); ++j) {
          if (new_relations[i][j]->relation_.count(node.name_)) {
            auto& curr_relation = new_relations[i][j]->relation_[node.name_];
            for (auto&& curr_result_ref : curr_relation) {
              relations[i].push_back(&curr_result_ref);
            }
          }
        }
      }
    }
    std::vector<std::vector<std::string>> prefix(texts.size());
    for (int i = 0; i < input_mapping.size(); ++i) {
      auto&& input_mapping_item = input_mapping[i];
      for (auto&& idx : input_mapping_item) {
        for (int j = 0; j < results_list[idx].size(); ++j) {
          auto prefix_str = results_list[idx][j].text_ + "\xe7\x9a\x84";
          prefix[i].push_back(prefix_str);
        }
      }
    }
    for (auto& node_child : node.children_) {
      node_child.relations_ = relations;
      node_child.prefix_ = prefix;
      nodes.push(node_child);
    }
  }
}
