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
* @file linear_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/linear.h"
#include "poros/util/test_util.h"

static void linear_test_helper(const std::string& graph_IR,
                            const std::vector<at::Tensor>& input_data,
                            const std::vector<size_t> replace_const_index) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::LinearConverter linearconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &linearconverter, 
                input_data, graph_output, poros_output, nullptr, "", replace_const_index));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

static std::string gen_no_bias_graph() {
    std::string graph = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : None = prim::Constant()
          %3 : Tensor = aten::linear(%0, %1, %2)
          return (%3))IR";
    return graph;
}

TEST(Converters, ATenLinearNoBiasConvertsCorrectly) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    const auto graph_IR = gen_no_bias_graph();
    baidu::mirana::poros::LinearConverter linearconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 2}, {at::kCUDA}));
    input_data.push_back(at::randn({3, 2}, {at::kCUDA})); // 内部转置
    linear_test_helper(graph_IR, input_data, {});
}

TEST(Converters, ATenLinearNoBiasNeedPaddingConvertsCorrectly) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    const auto graph_IR = gen_no_bias_graph();
    baidu::mirana::poros::LinearConverter linearconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 64, 8}, {at::kCUDA}));
    input_data.push_back(at::randn({30, 8}, {at::kCUDA})); // 内部转置
    linear_test_helper(graph_IR, input_data, {});
}

TEST(Converters, ATenLinearNoBiasNeedPaddingConstWeightConvertsCorrectly) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    const auto graph_IR = gen_no_bias_graph();
    baidu::mirana::poros::LinearConverter linearconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 64, 8}, {at::kCUDA}));
    input_data.push_back(at::randn({30, 8}, {at::kCUDA})); // 内部转置
    linear_test_helper(graph_IR, input_data, {1}); //把第二个参数转换成常量
}

TEST(Converters, ATenLinearNoBiasNeedPaddingConstWeight2ConvertsCorrectly) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    const auto graph_IR = gen_no_bias_graph();
    baidu::mirana::poros::LinearConverter linearconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 64, 64, 8}, {at::kCUDA}));
    input_data.push_back(at::randn({30, 8}, {at::kCUDA})); // 内部转置
    linear_test_helper(graph_IR, input_data, {1});  //把第二个参数转换成常量
}

TEST(Converters, ATenLinearBiasConvertsCorrectly) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor, %2 : Tensor):
          %3 : Tensor = aten::linear(%0, %1, %2)
          return (%3))IR";
    baidu::mirana::poros::LinearConverter linearconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 3}, {at::kCUDA}));
    input_data.push_back(at::randn({2, 3}, {at::kCUDA}));
    input_data.push_back(at::randn({2}, {at::kCUDA}));
    linear_test_helper(graph_IR, input_data, {});
}