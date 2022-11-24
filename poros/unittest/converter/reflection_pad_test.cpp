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
* @file reflection_pad_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/reflection_pad.h"
#include "poros/util/test_util.h"

static void reflection_pad_test_helper(const std::string& graph_IR, 
                                std::vector<int64_t> shape,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));

    baidu::mirana::poros::ReflectionPadConverter reflectionpadconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &reflectionpadconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_reflection_pad_graph(const std::string& op, 
                                            const std::string& padding) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[)IR" + padding + R"IR(]]()
        %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
        return (%2))IR";
}

TEST(Converters, ATenReflectionPad1DConvertsCorrectly) {
    // aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
    const auto graph_IR = gen_reflection_pad_graph("reflection_pad1d", "2, 2");
    reflection_pad_test_helper(graph_IR, {2, 5});
}

TEST(Converters, ATenReflectionPad2DConvertsCorrectly) {
    // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = gen_reflection_pad_graph("reflection_pad2d", "1, 1, 2, 3");
    reflection_pad_test_helper(graph_IR, {3, 4, 3});
}

TEST(Converters, ATenReflectionPad1DDynamicConvertsCorrectly) {
    // aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
    const auto graph_IR = gen_reflection_pad_graph("reflection_pad1d", "2, 3");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({3, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 5}, {at::kCUDA}));

    reflection_pad_test_helper(graph_IR, {2, 5}, true, &prewarm_data);
}

TEST(Converters, ATenReflectionPad2DDynamicConvertsCorrectly) {
    // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = gen_reflection_pad_graph("reflection_pad2d", "1, 1, 2, 3");
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({3, 4, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({3, 4, 3}, {at::kCUDA}));

    reflection_pad_test_helper(graph_IR, {3, 4, 3}, true, &prewarm_data);
}

TEST(Converters, ATenReflectionPad1DDynamicscalarinputConvertsCorrectly) {
    // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : int = aten::size(%0, %1)
        %5 : float = aten::div(%4, %3)
        %6 : int = aten::floor(%5)
        %7 : int[] = prim::ListConstruct(%1, %6)
        %8 : Tensor = aten::reflection_pad1d(%0, %7)
        return (%8))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({3, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 5}, {at::kCUDA}));

    reflection_pad_test_helper(graph_IR, {2, 7}, true, &prewarm_data);
}


TEST(Converters, ATenReflectionPad2DDynamicscalarinputConvertsCorrectly) {
    // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : int = aten::size(%0, %1)
        %5 : float = aten::div(%4, %3)
        %6 : int = aten::floor(%5)
        %7 : int[] = prim::ListConstruct(%1, %2, %3, %6)
        %8 : Tensor = aten::reflection_pad2d(%0, %7)
        return (%8))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 7, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({3, 5, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({3, 5, 3}, {at::kCUDA}));

    reflection_pad_test_helper(graph_IR, {3, 5, 3}, true, &prewarm_data);

}