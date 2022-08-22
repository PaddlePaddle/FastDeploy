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
#include "uie.h"

using namespace paddlenlp;

int main() {
  auto predictor =
      UIEModel("uie-base/inference.pdmodel", "uie-base/inference.pdiparams",
               "uie-base/vocab.txt", 0.5, 128, {"时间", "选手", "赛事名称"});
  fastdeploy::FDINFO << "After init predictor" << std::endl;
  std::vector<std::unordered_map<std::string, std::vector<UIEResult>>> results;
  predictor.Predict({"2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷"
                     "爱凌以188.25分获得金牌！"},
                    &results);
  return 0;
}
