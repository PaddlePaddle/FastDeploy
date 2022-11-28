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
* @file squeeze_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/squeeze.h"
#include "poros/util/test_util.h"

static void squeeze_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                std::vector<int64_t> shape){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_squeeze_one_input_schema_graph(const std::string& op) {
    return R"IR(
        graph(%0 : Tensor):
          %2 : Tensor = aten::)IR" + op + R"IR((%0)
          %3 : Tensor = aten::relu(%2)
          return (%3))IR";
}

TEST(Converters, ATenSqueezeOneInputConvertsCorrectly) {
    // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_one_input_schema_graph("squeeze");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 1, 3});
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 1, 1, 5});
}

static std::string gen_squeeze_graph(const std::string& op, const std::string& dim) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          %3 : Tensor = aten::relu(%2)
          return (%3))IR";
}

TEST(Converters, ATenSqueezeConvertsCorrectly) {
    // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("squeeze", "1");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 1, 3});
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 2, 3});
}

TEST(Converters, ATenSqueezeNegtiveConvertsCorrectly) {
    // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("squeeze", "-1");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 3, 1});
    squeeze_test_helper(graph_IR, &squeezeconverter, {4, 2, 3});
}

TEST(Converters, ATenUnSqueezeConvertsCorrectly) {
    // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("unsqueeze", "1");
    baidu::mirana::poros::UnSqueezeConverter unsqueezeconverter;
    squeeze_test_helper(graph_IR, &unsqueezeconverter, {4, 3, 2});
}

TEST(Converters, ATenUnSqueezeNegtiveConvertsCorrectly) {
    // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("unsqueeze", "-1");
    baidu::mirana::poros::UnSqueezeConverter unsqueezeconverter;
    squeeze_test_helper(graph_IR, &unsqueezeconverter, {4, 3, 2});
}

static void squeeze_dy_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenSqueezeOneInputDynamicConvertsCorrectly) {
    // aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_one_input_schema_graph("squeeze");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({40, 1, 1, 60}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({20, 1, 1, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20, 1, 1, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({20, 1, 1, 40}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &squeezeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenUnSqueezeDynamicConvertsCorrectly) {
    // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("unsqueeze", "2");
    baidu::mirana::poros::UnSqueezeConverter unsqueezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({40, 50, 60}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &unsqueezeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenUnSqueezeInputSingleDimDynamicConvertsCorrectly) {
    // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("unsqueeze", "0");
    baidu::mirana::poros::UnSqueezeConverter unsqueezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({40}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({20}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({20}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &unsqueezeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenUnSqueezeDynamicNegtiveDimConvertsCorrectly) {
     // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("unsqueeze", "-1");
    baidu::mirana::poros::UnSqueezeConverter unsqueezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({40, 50, 60}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({20, 30, 40}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &unsqueezeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenSqueezeDynamicConvertsCorrectly) {
    // aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("squeeze", "1");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({40, 1, 60}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({20, 1, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20, 1, 40}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({20, 1, 40}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &squeezeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenSqueezeDynamicNegtiveDimConvertsCorrectly) {
    // aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)
    const auto graph_IR = gen_squeeze_graph("squeeze", "-1");
    baidu::mirana::poros::SqueezeConverter squeezeconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    prewarm_data[0].push_back(at::randn({1, 60, 1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({1, 40, 1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({1, 40, 1}, {at::kCUDA}));
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 40, 1}, {at::kCUDA}));
    
    squeeze_dy_test_helper(graph_IR, &squeezeconverter, input_data, true, &prewarm_data);
}