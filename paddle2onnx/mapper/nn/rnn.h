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
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class RnnMapper : public Mapper {
 public:
  RnnMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    MarkAsExperimentalOp();
    GetAttr("num_layers", &num_layers_);
    GetAttr("input_size", &input_size_);
    GetAttr("hidden_size", &hidden_size_);
    GetAttr("seed", &seed_);
    GetAttr("dropout_prob", &dropout_prob_);
    GetAttr("mode", &mode_);
    GetAttr("is_bidirec", &is_bidirec_);
    if (HasAttr("is_test")) {
      GetAttr("is_test", &is_test_);
    }
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();

 private:
  std::vector<std::string> MakeParamInputs(int64_t layer_index);
  std::vector<std::string> MakeInitParamInputs(int64_t layer_index);
  std::string ReformWeight(const std::string& weight, const int64_t& size, const std::vector<int64_t>& perm);
  int64_t num_layers_;
  int64_t input_size_;
  int64_t hidden_size_;
  int64_t seed_;
  float dropout_prob_;
  std::string mode_;
  bool is_test_ = false;
  bool is_bidirec_;
};

}  // namespace paddle2onnx
