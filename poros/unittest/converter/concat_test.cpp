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
* @file concat_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/concat.h"
#include "poros/util/test_util.h"

static void cat_test_helper(const std::string& graph_IR, 
                                std::vector<int64_t> shape1 = {5}, 
                                std::vector<int64_t> shape2 = {5},
                                bool Triple_inputs = false,
                                std::vector<int64_t> shape3 = {5}){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    input_data.push_back(at::randn(shape2, {at::kCUDA}));
    if (Triple_inputs){
        input_data.push_back(at::randn(shape3, {at::kCUDA}));
    }
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::ConcatConverter concatconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &concatconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_double_inputs_cat_graph(const std::string& dim) {
    return R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : Tensor[] = prim::ListConstruct(%0, %1)
          %3 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %4 : Tensor = aten::cat(%2, %3)
          return (%4))IR";
}

static std::string gen_triple_inputs_cat_graph(const std::string& dim) {
    return R"IR(
        graph(%0 : Tensor, %1 : Tensor, %2 : Tensor):
          %3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          %4 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %5 : Tensor = aten::cat(%3, %4)
          return (%5))IR";
}

TEST(Converters, ATenCatPureTensorConvertsCorrectly) {
    // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
    const auto graph_IR = gen_double_inputs_cat_graph("0");
    cat_test_helper(graph_IR);
}

TEST(Converters, ATenCatPureTensorNegDimConvertsCorrectly) {
    // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
    const auto graph_IR = gen_double_inputs_cat_graph("-1");
    cat_test_helper(graph_IR, {5, 3}, {5, 4});
}

TEST(Converters, ATenCatTripleTensorConvertsCorrectly) {
    // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
    const auto graph_IR = gen_triple_inputs_cat_graph("1");
    cat_test_helper(graph_IR, {5, 2, 2}, {5, 7, 2}, true, {5, 3, 2});
}

TEST(Converters, ATenCatTripleTensorNegdimConvertsCorrectly) {
    // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
    const auto graph_IR = gen_triple_inputs_cat_graph("-1");
    cat_test_helper(graph_IR, {5, 6, 7}, {5, 6, 3}, true, {5, 6, 5});
}