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
#include "faster_tokenizer/tokenizers/ernie_faster_tokenizer.h"
#include "uie.h"

using namespace paddlenlp;

int main() {
  auto predictor =
      UIEModel("uie-base/inference.pdmodel", "uie-base/inference.pdiparams",
               "uie-base/vocab.txt", 0.5, 128, {"时间", "选手", "赛事名称"});
  fastdeploy::FDINFO << "After init predictor" << std::endl;
  std::vector<std::unordered_map<std::string, std::vector<UIEResult>>> results;
  // Named Entity Recognition
  predictor.Predict({"2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷"
                     "爱凌以188.25分获得金牌！"},
                    &results);
  std::cout << results << std::endl;
  results.clear();

  // Relation Extraction
  predictor.SetSchema({{"竞赛名称",
                        {SchemaNode("主办方"), SchemaNode("承办方"),
                         SchemaNode("已举办次数")}}});
  predictor.Predict(
      {"2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度"
       "公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会"
       "承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"},
      &results);
  std::cout << results << std::endl;
  results.clear();

  // Event Extraction
  predictor.SetSchema({{"地震触发词",
                        {SchemaNode("地震强度"), SchemaNode("时间"),
                         SchemaNode("震中位置"), SchemaNode("震源深度")}}});
  predictor.Predict(
      {"中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24."
       "34度，东经99.98度)发生3.5级地震，震源深度10千米。"},
      &results);
  std::cout << results << std::endl;
  results.clear();

  // Opinion Extraction
  predictor.SetSchema(
      {{"评价维度",
        {SchemaNode("观点词"), SchemaNode("情感倾向[正向，负向]")}}});
  predictor.Predict(
      {"店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"},
      &results);
  std::cout << results << std::endl;
  results.clear();

  // Sequence classification
  predictor.SetSchema({"情感倾向[正向，负向]"});
  predictor.Predict({"这个产品用起来真的很流畅，我非常喜欢"}, &results);
  std::cout << results << std::endl;
  results.clear();

  // Cross task extraction

  predictor.SetSchema({{"法院", {}},
                       {"原告", {SchemaNode("委托代理人")}},
                       {"被告", {SchemaNode("委托代理人")}}});
  predictor.Predict({"北京市海淀区人民法院\n民事判决书\n(199x)"
                     "建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 "
                     "A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司"
                     "总经理。\n委托代理人赵六，北京市 C律师事务所律师。"},
                    &results);
  std::cout << results << std::endl;
  results.clear();
  return 0;
}
