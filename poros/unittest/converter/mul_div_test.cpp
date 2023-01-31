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
* @file mul_div_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/mul_div.h"
#include "poros/util/test_util.h"

static void mul_div_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter, 
                                bool singleInput,
                                std::vector<int64_t> shape1 = {5}, 
                                std::vector<int64_t> shape2 = {5}) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    if (!singleInput){
        input_data.push_back(at::randn(shape2, {at::kCUDA}));
    }
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

std::string gen_mul_div_tensor_graph(const std::string& op) {
    return R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
        return (%2))IR";
}

std::string gen_mul_div_scalar_graph(const std::string& op, const std::string& scalar) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=)IR" + scalar + R"IR(]()
        %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
        return (%2))IR";
}

TEST(Converters, ATenMulConvertsCorrectly) {
    // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_mul_div_tensor_graph("mul");
    baidu::mirana::poros::MulConverter mulconverter;
    mul_div_test_helper(graph_IR, &mulconverter, false);
    mul_div_test_helper(graph_IR, &mulconverter, false, {3, 4}, {4});
    mul_div_test_helper(graph_IR, &mulconverter, false, {4}, {3, 4});
    mul_div_test_helper(graph_IR, &mulconverter, false, {4, 1}, {1, 4});
    mul_div_test_helper(graph_IR, &mulconverter, false, {3, 4, 3}, {4, 3});
    mul_div_test_helper(graph_IR, &mulconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenMulScalarConvertsCorrectly) {
    // aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
    const auto graph_IR = gen_mul_div_scalar_graph("mul", "2.4");
    baidu::mirana::poros::MulConverter mulconverter;
    mul_div_test_helper(graph_IR, &mulconverter, true);
    mul_div_test_helper(graph_IR, &mulconverter, true, {3, 4, 3});
}

TEST(Converters, ATenMul_ConvertsCorrectly) {
    // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
    const auto graph_IR = gen_mul_div_tensor_graph("mul_");
    baidu::mirana::poros::MulConverter mulconverter;
    mul_div_test_helper(graph_IR, &mulconverter, false);
    mul_div_test_helper(graph_IR, &mulconverter, false, {3, 4}, {4});
    mul_div_test_helper(graph_IR, &mulconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenMul_ScalarConvertsCorrectly) {
    // aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
    const auto graph_IR = gen_mul_div_scalar_graph("mul_", "2.4");
    baidu::mirana::poros::MulConverter mulconverter;
    mul_div_test_helper(graph_IR, &mulconverter, true);
    mul_div_test_helper(graph_IR, &mulconverter, true, {3, 4, 3});
}

TEST(Converters, ATenDivConvertsCorrectly) {
    // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_mul_div_tensor_graph("div");
    baidu::mirana::poros::DivConverter divconverter;
    mul_div_test_helper(graph_IR, &divconverter, false);
    mul_div_test_helper(graph_IR, &divconverter, false, {3, 4}, {4});
    mul_div_test_helper(graph_IR, &divconverter, false, {4}, {3, 4});
    mul_div_test_helper(graph_IR, &divconverter, false, {4, 1}, {1, 4});
    mul_div_test_helper(graph_IR, &divconverter, false, {3, 4, 3}, {4, 3});
    mul_div_test_helper(graph_IR, &divconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenDivScalarConvertsCorrectly) {
    // aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)
    const auto graph_IR = gen_mul_div_scalar_graph("div", "2.4");
    baidu::mirana::poros::DivConverter divconverter;
    mul_div_test_helper(graph_IR, &divconverter, true);
    mul_div_test_helper(graph_IR, &divconverter, true, {3, 4, 3});
}

TEST(Converters, ATenDiv_ConvertsCorrectly) {
    // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
    const auto graph_IR = gen_mul_div_tensor_graph("div_");
    baidu::mirana::poros::DivConverter divconverter;
    mul_div_test_helper(graph_IR, &divconverter, false);
    mul_div_test_helper(graph_IR, &divconverter, false, {3, 4}, {4});
    mul_div_test_helper(graph_IR, &divconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenDiv_ScalarConvertsCorrectly) {
    // aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
    const auto graph_IR = gen_mul_div_scalar_graph("div_", "2.4");
    baidu::mirana::poros::DivConverter divconverter;
    mul_div_test_helper(graph_IR, &divconverter, true);
    mul_div_test_helper(graph_IR, &divconverter, true, {3, 4, 3});
}

TEST(Converters, ATenDivIntDivideIntConvertsCorrectly) {
    // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_mul_div_tensor_graph("div");

    auto options_pyt_int = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    std::vector<at::Tensor> input_data;
    input_data.push_back(torch::tensor({14}, options_pyt_int));
    input_data.push_back(torch::tensor({2}, options_pyt_int));

    baidu::mirana::poros::DivConverter divconverter;
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &divconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenDivFloatDivideIntConvertsCorrectly) {
    // aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %1 : int = prim::Constant[value=3]()
            %2 : Tensor = aten::div(%0, %1)
            return (%2))IR";

    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    std::vector<at::Tensor> input_data;
    input_data.push_back(torch::tensor({15.3}, options_pyt_float));

    baidu::mirana::poros::DivConverter divconverter;
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &divconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenDivIntDivideFloatConvertsCorrectly) {
    // aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)
    const auto graph_IR = gen_mul_div_scalar_graph("div", "2.4");

    auto options_pyt_int = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    std::vector<at::Tensor> input_data;
    input_data.push_back(torch::tensor({15}, options_pyt_int));

    baidu::mirana::poros::DivConverter divconverter;
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &divconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenRemainderConvertsCorrectly) {
    // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_mul_div_tensor_graph("remainder");
    baidu::mirana::poros::RemainderConverter remainder;
    mul_div_test_helper(graph_IR, &remainder, false);
    mul_div_test_helper(graph_IR, &remainder, false, {3, 4}, {4});
    mul_div_test_helper(graph_IR, &remainder, false, {4}, {3, 4});
    mul_div_test_helper(graph_IR, &remainder, false, {4, 1}, {1, 4});
    mul_div_test_helper(graph_IR, &remainder, false, {3, 4, 3}, {4, 3});
    mul_div_test_helper(graph_IR, &remainder, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenRemainderScalarConvertsCorrectly) {
    // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
    const auto graph_IR = gen_mul_div_scalar_graph("remainder", "-0.4");
    baidu::mirana::poros::RemainderConverter remainder;
    mul_div_test_helper(graph_IR, &remainder, true);
    mul_div_test_helper(graph_IR, &remainder, true, {3, 4, 3});
}


static void mul_div_dynamic_test_helper(const std::string& graph_IR, 
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
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenMulIntdynamicConvertsCorrectly) {
    // aten::mul.int(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int = aten::mul(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::MulConverter mulconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    mul_div_dynamic_test_helper(graph_IR, &mulconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenDivIntdynamicConvertsCorrectly) {
    // aten::div.int(int a, int b) -> (float)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : float = aten::div(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::DivConverter divconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({4, 5}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({10, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::zeros({4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::zeros({4, 5}, {at::kCUDA}));

    mul_div_dynamic_test_helper(graph_IR, &divconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenDivNegIntdynamicConvertsCorrectly) {
    // aten::div.int(int a, int b) -> (float)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %34 : int = prim::Constant[value=100]()
          %35 : int = aten::sub(%3, %34)
          %5 : float = aten::div(%35, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::DivConverter divconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({4, 5}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({10, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::zeros({4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::zeros({4, 5}, {at::kCUDA}));

    mul_div_dynamic_test_helper(graph_IR, &divconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenFloordivIntdynamicConvertsCorrectly) {
    // aten::floordiv.int(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int = aten::floordiv(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::FloordivConverter floordivconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({12, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    mul_div_dynamic_test_helper(graph_IR, &floordivconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenFloordivNegIntdynamicConvertsCorrectly) {
    // aten::floordiv.int(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %34 : int = prim::Constant[value=100]()
          %35 : int = aten::sub(%3, %34)
          %5 : int = aten::floordiv(%35, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::FloordivConverter floordivconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({12, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    mul_div_dynamic_test_helper(graph_IR, &floordivconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenRoundToZeroFloordivIntdynamicConvertsCorrectly) {
    // aten::__round_to_zero_floordiv(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int = aten::__round_to_zero_floordiv(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::FloordivConverter floordivconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({12, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    mul_div_dynamic_test_helper(graph_IR, &floordivconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenRoundToZeroFloordivNegIntdynamicConvertsCorrectly) {
    // aten::__round_to_zero_floordiv(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %34 : int = prim::Constant[value=100]()
          %35 : int = aten::sub(%3, %34)
          %5 : int = aten::__round_to_zero_floordiv(%35, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::FloordivConverter floordivconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({12, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({10, 4}, {at::kCUDA}).to(at::ScalarType::Int));

    mul_div_dynamic_test_helper(graph_IR, &floordivconverter, input_data, true, &prewarm_data);
}