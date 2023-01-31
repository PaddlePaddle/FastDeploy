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
* @file logical_test.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-17 18:32:15
* @brief
**/

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/logical.h"
#include "poros/util/test_util.h"

enum InputTypeEnum {
    TYPE_A = 0, //  [4]*[4]
    TYPE_B, // [2,2]*[2,2]
    TYPE_C, // [4]*[true]
    TYPE_D, //broadcasting [1,3,2]*[2]
    TYPE_E, //broadcasting [2,3,4]*[3,4]
};

static std::vector<at::Tensor> get_input_data(const InputTypeEnum input_type) {
    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBool);

    switch (input_type) {
        case TYPE_A: //  [4]*[4]
            input_data.push_back(torch::tensor({false, true, false, true}, options_pyt));
            input_data.push_back(torch::tensor({false, true, true, true}, options_pyt));
            break;
        case TYPE_B:// [2,2]*[2,2]
            input_data.push_back(torch::tensor({{false, true},
                                                {false, true}}, options_pyt));
            input_data.push_back(torch::tensor({{false, true},
                                                {true,  true}}, options_pyt));
            break;
        case TYPE_C:// [4]*[1]
            input_data.push_back(torch::tensor({false, true, false, true}, options_pyt));
            input_data.push_back(torch::tensor({true}, options_pyt));
            break;
        case TYPE_D://broadcasting [1,3,2]*[2]
            input_data.push_back(torch::tensor({{{true, true}, {false, true}, {false, false}}}, options_pyt));
            input_data.push_back(torch::tensor({false, true}, options_pyt));
            break;
        case TYPE_E://broadcasting [2,3,4]*[3,4]
            input_data.push_back(torch::tensor({
                                                       {{false, true, false, true},  {false, true, false, false},
                                                               {true,  true, true,  true}},
                                                       {{false, true, false, false}, {true,  true, true,  true},
                                                               {false, true, false, true}}
                                               }, options_pyt));
            input_data.push_back(torch::tensor({{false, true, false, true},
                                                {false, true, false, false},
                                                {true,  true, true,  true}}, options_pyt));
            break;
    }

    return input_data;
}

static void and_test_helper(const std::string &graph_IR,
                            baidu::mirana::poros::IConverter *converter,
                            const InputTypeEnum input_type) {

    auto input_data = get_input_data(input_type);
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter,
                                                                    input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

static std::string gen_and_or_tensor_graph(const std::string &op) {
    return R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          return (%2))IR";
}

static std::string gen_not_tensor_graph(const std::string &op) {
    return R"IR(
        graph(%0 : Tensor):
          %2 : Tensor = aten::)IR" + op + R"IR((%0)
          return (%2))IR";
}

TEST(Converters, ATenLogicalAndConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("__and__");
    baidu::mirana::poros::AndConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

TEST(Converters, ATenLogicalBitwiseAndConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("bitwise_and");
    baidu::mirana::poros::AndConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

TEST(Converters, ATenLogicalOrConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("__or__");
    baidu::mirana::poros::OrConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

TEST(Converters, ATenLogicalBitwiseOrConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("bitwise_or");
    baidu::mirana::poros::OrConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

TEST(Converters, ATenLogicalXOrConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("__xor__");
    baidu::mirana::poros::XorConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

TEST(Converters, ATenLogicalBitwiseXOrConvertsCorrectly) {

    const auto graph_IR = gen_and_or_tensor_graph("bitwise_xor");
    baidu::mirana::poros::XorConverter converter;
    and_test_helper(graph_IR, &converter, TYPE_A);
    and_test_helper(graph_IR, &converter, TYPE_B);
    and_test_helper(graph_IR, &converter, TYPE_C);
    and_test_helper(graph_IR, &converter, TYPE_D);
    and_test_helper(graph_IR, &converter, TYPE_E);
}

static void not_test_helper(const std::string &graph_IR,
                            baidu::mirana::poros::IConverter *converter,
                            const InputTypeEnum input_type) {

    auto input_data = get_input_data(input_type);
    input_data.pop_back(); // only need one input, pop out the last one
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter,
                                                                    input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLogicalBitwiseNotConvertsCorrectly) {

    const auto graph_IR = gen_not_tensor_graph("bitwise_not");
    baidu::mirana::poros::NotConverter converter;
    not_test_helper(graph_IR, &converter, TYPE_A);
    not_test_helper(graph_IR, &converter, TYPE_B);
    not_test_helper(graph_IR, &converter, TYPE_C);
    not_test_helper(graph_IR, &converter, TYPE_D);
    not_test_helper(graph_IR, &converter, TYPE_E);
}

//}