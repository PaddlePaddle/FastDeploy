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
* @file unary_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/unary.h"
#include "poros/util/test_util.h"

static void unary_test_helper(const std::string& op,
                            std::vector<int64_t> shape = {10}){
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor = aten::)IR" +op + R"IR((%0)
        return (%1))IR";
    std::vector<at::Tensor> input_data;
    float offset = 0;
    if(op == "acosh"){
        offset += 1;
    }
    if(op == "abs" || op == "neg"){
        offset -= 0.5;
    }
    auto input_tensor = at::empty(shape, {at::kCUDA}).uniform_(0 + offset, 0.5 + offset); 
    if(op == "round") {
        input_tensor = input_tensor * 50;
    }
    input_data.push_back(input_tensor);
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::UnaryConverter unaryconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &unaryconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    // ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenCosConvertsCorrectly) {
    // aten::cos(Tensor self) -> Tensor
    unary_test_helper("cos");
}

TEST(Converters, ATenAcosConvertsCorrectly) {
    // aten::acos(Tensor self) -> Tensor
    unary_test_helper("acos");
}

TEST(Converters, ATenCoshConvertsCorrectly) {
    // aten::cosh(Tensor self) -> Tensor
    unary_test_helper("cosh");
}

TEST(Converters, ATenSinConvertsCorrectly) {
    // aten::sin(Tensor self) -> Tensor
    unary_test_helper("sin");
}

TEST(Converters, ATenAsinConvertsCorrectly) {
    // aten::asin(Tensor self) -> Tensor
    unary_test_helper("asin");
}

TEST(Converters, ATenSinhConvertsCorrectly) {
    // aten::sinh(Tensor self) -> Tensor
    unary_test_helper("sinh");
}

TEST(Converters, ATenTanConvertsCorrectly) {
    // aten::tan(Tensor self) -> Tensor
    unary_test_helper("tan");
}

TEST(Converters, ATenAtanConvertsCorrectly) {
    // aten::atan(Tensor self) -> Tensor
    unary_test_helper("atan");
}

TEST(Converters, ATenAbsConvertsCorrectly) {
    // aten::abs(Tensor self) -> Tensor
    unary_test_helper("abs");
}

TEST(Converters, ATenFloorConvertsCorrectly) {
    // aten::floor(Tensor self) -> Tensor
    unary_test_helper("floor");
}

TEST(Converters, ATenReciprocalConvertsCorrectly) {
    // aten::reciprocal(Tensor self) -> Tensor
    unary_test_helper("reciprocal");
}

TEST(Converters, ATenLogConvertsCorrectly) {
    // aten::log(Tensor self) -> Tensor
    unary_test_helper("log");
}

TEST(Converters, ATenCeilConvertsCorrectly) {
    // aten::ceil(Tensor self) -> Tensor
    unary_test_helper("ceil");
}

TEST(Converters, ATenSqrtConvertsCorrectly) {
    // aten::sqrt(Tensor self) -> Tensor
    unary_test_helper("sqrt");
}

TEST(Converters, ATenExpConvertsCorrectly) {
    // aten::exp(Tensor self) -> Tensor
    unary_test_helper("exp");
}

TEST(Converters, ATenNegConvertsCorrectly) {
    // aten::neg(Tensor self) -> Tensor
    unary_test_helper("neg");
}

TEST(Converters, ATenErfConvertsCorrectly) {
    // aten::erf(Tensor self) -> Tensor
    unary_test_helper("erf");
}

TEST(Converters, ATenAsinhConvertsCorrectly) {
    // aten::asinh(Tensor self) -> Tensor
    unary_test_helper("asinh");
}

TEST(Converters, ATenAcoshConvertsCorrectly) {
    // aten::acosh(Tensor self) -> Tensor
    unary_test_helper("acosh");
}

TEST(Converters, ATenAtanhConvertsCorrectly) {
    // aten::atanh(Tensor self) -> Tensor
    unary_test_helper("atanh");
}

TEST(Converters, ATenLog2ConvertsCorrectly) {
    // aten::log2(Tensor self) -> Tensor
    unary_test_helper("log2");
}

TEST(Converters, ATenLog10ConvertsCorrectly) {
    // aten::log10(Tensor self) -> Tensor
    unary_test_helper("log10");
}

TEST(Converters, ATenRoundConvertsCorrectly) {
    // aten::round(Tensor self) -> (Tensor)
    unary_test_helper("round");
}

TEST(Converters, ATenFloorFloat2IntConvertsCorrectly) {
    // aten::floor.float(float a) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %dim0 : int = prim::Constant[value=0]()
            %dim1 : int = prim::Constant[value=1]()
            %1 : float = prim::Constant[value=-1.5]()
            %2 : int = aten::size(%0, %dim0)
            %3 : int = aten::size(%0, %dim1)
            %4 : float = aten::div(%2, %3)
            %5 : int = aten::floor(%4)
            %6 : int = aten::floor(%1)
            %7 : int[] = prim::ListConstruct(%5, %6)
            %8 : NoneType = prim::Constant()
            %9 : bool = prim::Constant[value=0]()
            %10 : Device = prim::Constant[value="cuda:0"]()
            %11 : Tensor = aten::tensor(%7, %8, %10, %9)
            return (%11))IR";

    baidu::mirana::poros::UnaryConverter unaryconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({7, 2}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({3, 2}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({5, 2}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::ones({7, 2}, {at::kCUDA}));

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &unaryconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}