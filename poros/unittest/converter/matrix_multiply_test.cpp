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
* @file matrix_multiply_test.cpp
* @author tianjinjin@baidu.com
* @date Tue Sep 14 18:19:00 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/matrix_multiply.h"
#include "poros/util/test_util.h"

static void matrix_multiply_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter, 
                                std::vector<int64_t> shape1, 
                                std::vector<int64_t> shape2,
                                bool tripleinputs = false,
                                std::vector<int64_t> shape3 = {5}) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    input_data.push_back(at::randn(shape2, {at::kCUDA}));
    if (tripleinputs){
        input_data.push_back(at::randn(shape3, {at::kCUDA}));
    }
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    c10::ShowLogInfoToStderr();
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenMatmulConvertersCorrectly) {
    // aten::matmul(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::matmul(%0, %1)
        return (%2))IR";
    baidu::mirana::poros::MatmulConverter matmulconverter;
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {3}, {3});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {1, 1536}, {1536, 2});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {3}, {3, 512});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {512}, {512, 3});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {512, 3}, {3});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {1, 30, 1024}, {1024});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {1, 30, 1024}, {1024, 214});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {8}, {512, 8, 10});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {254, 8}, {512, 8, 10});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {10, 3, 512}, {10, 512, 214});
    matrix_multiply_test_helper(graph_IR, &matmulconverter, {10, 1, 24, 224}, {7, 224, 5});
}

TEST(Converters, ATenBmmConvertersCorrectly) {
    // aten::bmm(Tensor self, Tensor mat2) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::bmm(%0, %1)
        return (%2))IR";
    baidu::mirana::poros::BmmConverter bmmconverter;
    matrix_multiply_test_helper(graph_IR, &bmmconverter, {10, 3, 4}, {10, 4, 5});
}

static std::string gen_addmm_graph(const std::string& beta, const std::string& alpha) {
    return R"IR(
      graph(%0 : Tensor, %1 : Tensor, %2 : Tensor):
        %3 : float = prim::Constant[value=)IR" + beta + R"IR(]()
        %4 : float = prim::Constant[value=)IR" + alpha + R"IR(]()
        %5 : Tensor = aten::addmm(%0, %1, %2, %3, %4)
        return (%5))IR";
}

TEST(Converters, ATenAddmmConvertersCorrectly) {
    // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_addmm_graph("1.0", "1.0");
    baidu::mirana::poros::AddmmConverter addmmconverter;
    matrix_multiply_test_helper(graph_IR, &addmmconverter, {2, 3}, {2, 3}, true, {3, 3});
    matrix_multiply_test_helper(graph_IR, &addmmconverter, {3}, {2, 3}, true, {3, 3});
}

TEST(Converters, ATenAddmmBetaAlphaConvertersCorrectly) {
    // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_addmm_graph("3.3", "2.2");
    baidu::mirana::poros::AddmmConverter addmmconverter;
    matrix_multiply_test_helper(graph_IR, &addmmconverter, {2, 3}, {2, 3}, true, {3, 3});
}