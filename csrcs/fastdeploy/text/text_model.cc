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

#include "fastdeploy/text/text_model.h"

namespace fastdeploy {
namespace text {

bool TextModel::Predict(const std::string& raw_text, TextResult* result,
                        const PredictionOption& option) const {
  return true;
}

bool TextModel::PredictBatch(const std::vector<std::string>& raw_text_array,
                             std::vector<TextResult>* results,
                             const PredictionOption& option) const {
  return true;
}

}  // namespace text
}  // namespace fastdeploy