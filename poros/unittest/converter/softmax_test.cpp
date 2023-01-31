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
* @file softmax_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/softmax.h"
#include "poros/util/test_util.h"

static void softmax_test_helper(const std::string& graph_IR, 
                            std::vector<int64_t> shape = {5}){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    // input_data.push_back(at::randint(0, 5, {5}, {at::kCUDA}));
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::SoftmaxConverter softmaxconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &softmaxconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

static std::string gen_softmax_graph(const std::string& dim) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %2 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %3 : Tensor = aten::softmax(%0, %2, %1)
          return (%3))IR";
}

TEST(Converters, ATenSoftmax1DConvertsCorrectly) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("0");
    softmax_test_helper(graph_IR, {5});
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlySub3DIndex) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("1");
    softmax_test_helper(graph_IR, {1, 2, 3, 4, 5});
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlyAbove3DIndex) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("3");
    softmax_test_helper(graph_IR, {1, 2, 3, 4, 5});
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlyNegtiveOneIndex) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("-1");
    softmax_test_helper(graph_IR, {1, 2, 3, 4, 5});
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlyNegtiveIndex) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("-2");
    softmax_test_helper(graph_IR, {1, 2, 3, 4, 5});
}

static void softmax_dy_test_helper(const std::string& graph_IR, 
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    baidu::mirana::poros::SoftmaxConverter softmaxconverter;
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &softmaxconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenSoftmaxInputSingleDimDynamicConvertsCorrectly) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("0");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({60}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({40}, {at::kCUDA}));
    
    softmax_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, ATenSoftmaxDynamicConvertsCorrectly) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("2");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({20, 30, 40, 50}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    
    softmax_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, ATenSoftmaxDynamicNegtiveDimConvertsCorrectly) {
    // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_softmax_graph("-2");

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({20, 30, 40, 50}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 20, 30, 40}, {at::kCUDA}));
    
    softmax_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}