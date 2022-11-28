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
* @file topk_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/topk.h"
#include "poros/util/test_util.h"

static void topk_test_helper(const std::string& graph_IR,
                            std::vector<int64_t> shape) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::TopkConverter topkconverter;
    
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &topkconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(2, graph_output.size());
    ASSERT_EQ(2, poros_output.size());
    
    // ASSERT_TRUE(baidu::mirana::poros::testutil::almostEqual(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
    ASSERT_TRUE(graph_output[1].equal(poros_output[1]));
}
static std::string gen_topk_graph(const std::string& k,
                                const std::string& dim,
                                const std::string& largest,
                                const std::string& sorted) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + k + R"IR(]()
          %2 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %3 : bool = prim::Constant[value=)IR" + largest + R"IR(]()
          %4 : bool = prim::Constant[value=)IR" + sorted + R"IR(]()
          %5 : Tensor, %6 : Tensor = aten::topk(%0, %1, %2, %3, %4)
          return (%5, %6))IR";
}

TEST(Converters, ATenTopkConvertsCorrectly) {
    // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_topk_graph("10", "0", "1", "1");
    topk_test_helper(graph_IR, {20, 10});
}

TEST(Converters, ATenTopkDimConvertsCorrectly) {
    // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_topk_graph("5", "1", "1", "1");
    topk_test_helper(graph_IR, {20, 10});
}

TEST(Converters, ATenTopkDimNegtiveConvertsCorrectly) {
    // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_topk_graph("5", "-1", "1", "1");
    topk_test_helper(graph_IR, {20, 10});
}

TEST(Converters, ATenTopklargestConvertsCorrectly) {
    // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_topk_graph("10", "0", "0", "1");
    topk_test_helper(graph_IR, {20, 10});
}

// sorted argument is not used in TensorRT for aten::topk