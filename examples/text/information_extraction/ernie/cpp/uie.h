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
#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/utils/unique_ptr.h"
#include "tokenizers/ernie_faster_tokenizer.h"

using namespace paddlenlp;

struct SchemaNode {
  std::string name_;
  std::vector<std::string> prefix_;
  std::vector<std::vector<std::unordered_map<std::string, std::string>>>
      relations_;
  std::vector<SchemaNode> children_;

  explicit SchemaNode(const std::string& name) : name_(name) {}
  void AddChild(const std::string& schema) { children_.emplace_back(schema); }
  void AddChild(const std::string& schema,
                const std::vector<std::string>& children) {
    SchemaNode schema_node(schema);
    for (auto& child : children) {
      schema_node.children_.emplace_back(child);
    }
    children_.emplace_back(schema_node);
  }
};

struct Schema {
  explicit Schema(const std::string& schema, const std::string& name = "root");
  explicit Schema(const std::vector<std::string>& schema_list,
                  const std::string& name = "root");
  explicit Schema(const std::unordered_map<
                      std::string, std::vector<std::string>>& schema_map,
                  const std::string& name = "root");

 private:
  void CreateRoot(const std::string& name);
  std::unique_ptr<SchemaNode> root_;
  friend class UIEModel;
};

struct UIEResult {
  size_t start_;
  size_t end_;
  double probability_;
  std::string text_;
  std::unique_ptr<UIEResult> relations_;
};

struct UIEInput {
  std::string text_;
  std::string prompt_;
};

struct UIEModel {
 public:
  UIEModel(const std::string& model_file, const std::string& params_file,
           const std::string& vocab_file, float position_prob,
           size_t max_length, const std::vector<std::string>& schema);
  UIEModel(
      const std::string& model_file, const std::string& params_file,
      const std::string& vocab_file, float position_prob, size_t max_length,
      const std::unordered_map<std::string, std::vector<std::string>>& schema);
  void SetSchema(const std::vector<std::string>& schema);
  void SetSchema(
      const std::unordered_map<std::string, std::vector<std::string>>& schema);

  void PredictUIEInput(const std::vector<std::string>& input_texts,
                       const std::vector<std::string>& prompts);
  void Predict(const std::vector<std::string>& texts,
               std::vector<UIEResult>* results);

 private:
  using IDX_PROB = std::pair<int64_t, float>;
  struct IdxProbCmp {
    bool operator()(const std::pair<IDX_PROB, IDX_PROB>& lhs,
                    const std::pair<IDX_PROB, IDX_PROB>& rhs) const;
  };
  using SPAN_SET = std::set<std::pair<IDX_PROB, IDX_PROB>, IdxProbCmp>;
  void AutoSplitter(
      const std::vector<std::string>& texts, size_t max_length,
      std::vector<std::string>* short_texts,
      std::unordered_map<size_t, std::vector<size_t>>* input_mapping);
  // Get idx of the last dimension in probability arrays, which is greater than
  // a limitation.
  void GetCandidateIdx(const float* probs, int64_t batch_size, int64_t seq_len,
                       std::vector<std::vector<IDX_PROB>>* candidate_idx_prob,
                       float threshold = 0.5) const;
  void GetSpan(const std::vector<IDX_PROB>& start_idx_prob,
               const std::vector<IDX_PROB>& end_idx_prob,
               SPAN_SET* span_set) const;
  void GetSpanIdxAndProbs(
      const SPAN_SET& span_set,
      const faster_tokenizer::core::Offset& offset_mapping,
      std::vector<faster_tokenizer::core::Offset>* span_idxs,
      std::vector<float>* probs) const;
  fastdeploy::RuntimeOption runtime_option_;
  fastdeploy::Runtime runtime_;
  std::unique_ptr<Schema> schema_;
  size_t max_length_;
  float position_prob_;
  faster_tokenizer::tokenizers_impl::ErnieFasterTokenizer tokenizer_;
};
