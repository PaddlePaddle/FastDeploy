// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
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

/**
* @file element_wise_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/element_wise.h"
#include "poros/util/test_util.h"

static void poros_test_helper(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            const std::vector<at::Tensor>& input_data){
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    std::string pow_node_name("aten::pow");
    if(converter->node_kind()[0].toQualString() == pow_node_name){
        ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    }else{
        ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
    }
}

static void pow_test_examples(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            bool singleInput,
                            std::vector<int64_t> shape1 = {5}, 
                            std::vector<int64_t> shape2 = {5}){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    if (!singleInput){
        input_data.push_back(at::randint(-5, 5, shape2, {at::kCUDA}));
    }
    poros_test_helper(graph_IR, converter, input_data);
}

TEST(Converters, ATenPowTensorConvertsCorrectly) {
    // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
    const auto graph_IR = R"IR(
       graph(%1 : Tensor, %2 : Tensor):
          %3 : Tensor = aten::pow(%1, %2)
          return (%3))IR";
    baidu::mirana::poros::PowOrFloordivideConverter poworfloordivideconverter;
    pow_test_examples(graph_IR, &poworfloordivideconverter, false);
    pow_test_examples(graph_IR, &poworfloordivideconverter, false, {3, 4}, {4});
    pow_test_examples(graph_IR, &poworfloordivideconverter, false, {4}, {3, 4});
    pow_test_examples(graph_IR, &poworfloordivideconverter, false, {3, 4, 3}, {4, 3});
    pow_test_examples(graph_IR, &poworfloordivideconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenPowScalarConvertsCorrectly) {
    // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
    const auto graph_IR = R"IR(
            graph(%1 : Tensor):
              %2 : float = prim::Constant[value=2.0]()
              %3 : Tensor = aten::pow(%1, %2)
              return (%3))IR";
    baidu::mirana::poros::PowOrFloordivideConverter poworfloordivideconverter;
    pow_test_examples(graph_IR, &poworfloordivideconverter, true);
    pow_test_examples(graph_IR, &poworfloordivideconverter, true, {3, 4});
}

static void elementwise_tensor_test_examples(const std::string& op,
                                            baidu::mirana::poros::IConverter* converter){
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          return (%2))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    poros_test_helper(graph_IR, converter, input_data);

    input_data.clear();
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data[0][0][0] = 2.5;
    input_data[1][0][0] = 2.5;
    poros_test_helper(graph_IR, converter, input_data);

    input_data.clear();
    input_data.push_back(at::randn({3, 4, 3}, {at::kCUDA}));
    input_data.push_back(at::randn({4, 3}, {at::kCUDA}));
    input_data[0][0][0][0] = 2.5;
    input_data[1][0][0] = 2.5;
    poros_test_helper(graph_IR, converter, input_data);

    input_data.clear();
    input_data.push_back(at::randn({4, 3}, {at::kCUDA}));
    input_data.push_back(at::randn({3, 4, 3}, {at::kCUDA}));
    input_data[0][0][0] = 2.5;
    input_data[1][0][0][0] = 2.5;
    poros_test_helper(graph_IR, converter, input_data);
}

static void elementwise_scalar_test_examples(const std::string& op,
                                            const std::string& scalar,
                                            baidu::mirana::poros::IConverter* converter){
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : float = prim::Constant[value=)IR" + scalar + R"IR(]()
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          return (%2))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    poros_test_helper(graph_IR, converter, input_data);

    input_data.clear();
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data[0][0][0] = 2.5;
    poros_test_helper(graph_IR, converter, input_data);

    input_data.clear();
    input_data.push_back(at::randn({1}, {at::kCUDA}));
    input_data[0][0] = 2.5;
    poros_test_helper(graph_IR, converter, input_data);
}

TEST(Converters, ATenEqualTensorConvertsCorrectly) {
    // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::EqualOrNotequalConverter equalorbotequalconverter;
    elementwise_tensor_test_examples("eq", &equalorbotequalconverter);
}

TEST(Converters, ATenEqualScalarConvertsCorrectly) {
    // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::EqualOrNotequalConverter equalorbotequalconverter;
    elementwise_scalar_test_examples("eq", "2.5",&equalorbotequalconverter);
}

TEST(Converters, ATenNotEqualTensorConvertsCorrectly) {
    // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::EqualOrNotequalConverter equalorbotequalconverter;
    elementwise_tensor_test_examples("ne", &equalorbotequalconverter);
}

TEST(Converters, ATenNotEqualScalarConvertsCorrectly) {
    // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::EqualOrNotequalConverter equalorbotequalconverter;
    elementwise_scalar_test_examples("ne", "2.5", &equalorbotequalconverter);
}

TEST(Converters, ATenGtTensorConvertsCorrectly) {
    // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_tensor_test_examples("gt", &greaterorlessconverter);
}

TEST(Converters, ATenGtScalarConvertsCorrectly) {
    // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_scalar_test_examples("gt", "2.5", &greaterorlessconverter);
}

TEST(Converters, ATenLtTensorConvertsCorrectly) {
    // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_tensor_test_examples("lt", &greaterorlessconverter);
}

TEST(Converters, ATenLtScalarConvertsCorrectly) {
    // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_scalar_test_examples("lt", "2.5", &greaterorlessconverter);
}

TEST(Converters, ATenGeTensorConvertsCorrectly) {
    // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_tensor_test_examples("ge", &greaterorlessconverter);
}

TEST(Converters, ATenGeScalarConvertsCorrectly) {
    // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_scalar_test_examples("ge", "2.5", &greaterorlessconverter);
}

TEST(Converters, ATenLeTensorConvertsCorrectly) {
    // aten::le.Tensor(Tensor self, Tensor other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_tensor_test_examples("le", &greaterorlessconverter);
}

TEST(Converters, ATenLeScalarConvertsCorrectly) {
    // aten::le.Scalar(Tensor self, Scalar other) -> Tensor
    baidu::mirana::poros::GreaterOrLessConverter greaterorlessconverter;
    elementwise_scalar_test_examples("le", "2.5", &greaterorlessconverter);
}

static std::string gen_clamp_graph(const std::string& op,
                                  const std::string& min_val,
                                  const std::string& max_val){
    if (op == "clamp"){
        std::string min_val_IR;
        std::string max_val_IR;
        if (min_val.empty()){
            min_val_IR = "None = prim::Constant()";
        }else{
            min_val_IR = "float = prim::Constant[value=" + min_val + "]()";
        }
        if (max_val.empty()){
            max_val_IR = "None = prim::Constant()";
        }else{
            max_val_IR = "float = prim::Constant[value=" + max_val + "]()";
        }
        return R"IR(
            graph(%0 : Tensor):
              %1 : )IR" + min_val_IR + R"IR(
              %2 : )IR" + max_val_IR + R"IR(
              %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
              return (%3))IR";
    }else if (op == "clamp_min"){
        return R"IR(
            graph(%0 : Tensor):
              %1 : float = prim::Constant[value=)IR" + min_val + R"IR(]()
              %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
              return (%2))IR";
    }else if (op == "clamp_max"){
        return R"IR(
            graph(%0 : Tensor):
              %1 : float = prim::Constant[value=)IR" + max_val + R"IR(]()
              %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
              return (%2))IR";
    }else{
        return "";
    }
}

TEST(Converters, ATenClampMinConvertsCorrectly) {
    // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp", "1.5", "");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}

TEST(Converters, ATenClampMaxConvertsCorrectly) {
    // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp", "", "0.5");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}

TEST(Converters, ATenClampMinMaxConvertsCorrectly) {
    // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp", "-0.5", "0.5");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}

TEST(Converters, ATenClampMaximumConvertsCorrectly) {
    // aten::clamp_max(Tensor self, Scalar max) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp_max", "", "0.5");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}

TEST(Converters, ATenClampMinimumConvertsCorrectly) {
    // aten::clamp_min(Tensor self, Scalar min) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp_min", "-0.5", "");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}

TEST(Converters, ATenClampMinGtMaxConvertsCorrectly) {
    // aten::clamp_min(Tensor self, Scalar min) -> Tensor
    const auto graph_IR = gen_clamp_graph("clamp", "0.5", "-0.5");
    baidu::mirana::poros::ClampConverter clampconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    poros_test_helper(graph_IR, &clampconverter, input_data);
}