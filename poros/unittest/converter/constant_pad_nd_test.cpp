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
* @file constant_pad_nd_test.cpp
* @author tianshaoqing@baidu.com
* @date Thur Dec 2 14:29:20 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/util/test_util.h"
#include "poros/converter/gpu/constant_pad_nd.h"

static void constant_pad_nd_test_helper(const std::string& graph_IR, 
                            std::vector<at::Tensor> input_data,
                            bool is_dynamic = false,
                            std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    baidu::mirana::poros::ConstantPadNdConverter constantpadndconverter;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &constantpadndconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));

}

static std::string gen_constant_pad_nd_graph(const std::string& padding_shape_str, 
                                            const std::string& value_str,
                                            const bool padding_value_is_int = false) {
    if (padding_value_is_int) {
        return R"IR(
            graph(%0 : Tensor):
                %1 : int[] = prim::Constant[value=[)IR" + padding_shape_str + R"IR(]]()
                %2 : int = prim::Constant[value=)IR" + value_str + R"IR(]()
                %3 : Tensor = aten::constant_pad_nd(%0, %1, %2)
                return (%3))IR";

    } else {
        return R"IR(
            graph(%0 : Tensor):
                %1 : int[] = prim::Constant[value=[)IR" + padding_shape_str + R"IR(]]()
                %2 : float = prim::Constant[value=)IR" + value_str + R"IR(]()
                %3 : Tensor = aten::constant_pad_nd(%0, %1, %2)
                return (%3))IR";
    }
   
}

TEST(Converters, TestAtenConstantPadNdCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2, 3, 4", "1.5");
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    constant_pad_nd_test_helper(graph_IR, input_data);
}

TEST(Converters, TestAtenConstantPadNdLastDimCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2", "1.5");
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    constant_pad_nd_test_helper(graph_IR, input_data);
}

TEST(Converters, TestAtenConstantPadNdZerosPaddingDimsCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("0, 1, 2, 0", "1.5");
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    constant_pad_nd_test_helper(graph_IR, input_data);
}

TEST(Converters, TestAtenConstantPadNdIntCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2, 3, 4", "1", true);
    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    input_data.push_back(at::randint(0, 10, {4, 5, 6, 7}, options_pyt));
    constant_pad_nd_test_helper(graph_IR, input_data);
}

TEST(Converters, TestAtenConstantPadNdInputSingleDimCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2", "1.5");
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({6}, {at::kCUDA}));
    constant_pad_nd_test_helper(graph_IR, input_data);
}

TEST(Converters, TestAtenConstantPadNdDynamicFloatCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2, 3, 4", "1.5");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({3, 4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));

    constant_pad_nd_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, TestAtenConstantPadNdDynamicFloatTwoPaddingDimsZerosCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("2, 0, 0, 2", "1.5");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({3, 4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4, 5}, {at::kCUDA}));

    constant_pad_nd_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, TestAtenConstantPadNdDynamicFloatSingleDimCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2", "1.5");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({10}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({5}, {at::kCUDA}));

    constant_pad_nd_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, TestAtenConstantPadNdDynamicIntCorrectly) {
    const auto graph_IR = gen_constant_pad_nd_graph("1, 2, 3, 4", "2", true);

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    prewarm_data[0].push_back(at::randint(0, 10, {3, 4, 5, 6}, options_pyt));
    prewarm_data[1].push_back(at::randint(0, 10, {2, 3, 4, 5}, options_pyt));
    prewarm_data[2].push_back(at::randint(0, 10, {2, 3, 4, 5}, options_pyt));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randint(0, 10, {2, 3, 4, 5}, {at::kCUDA}));

    constant_pad_nd_test_helper(graph_IR, input_data, true, &prewarm_data);
}