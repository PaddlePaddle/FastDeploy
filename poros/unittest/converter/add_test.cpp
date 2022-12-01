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
* @file add_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/add.h"
#include "poros/util/test_util.h"

static void add_test_helper(const std::string& graph_IR, 
                            baidu::mirana::poros::IConverter* converter,
                            bool singleInput,
                            std::vector<int64_t> shape1 = {5}, 
                            std::vector<int64_t> shape2 = {5}){
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

static std::string gen_add_sub_tensor_graph(const std::string& op, 
                                            const std::string& alpha) {
    return R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : float = prim::Constant[value=)IR" + alpha + R"IR(]()
          %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
          return (%3))IR";
}

static std::string gen_add_sub_scalar_graph(const std::string& op, 
                                            const std::string& scalar,
                                            const std::string& alpha) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : float = prim::Constant[value=)IR" + scalar + R"IR(]()
          %2 : float = prim::Constant[value=)IR" + alpha + R"IR(]()
          %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
          return (%3))IR";
}

TEST(Converters, ATenAddTensorConvertsCorrectly) {
    // aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, false);
    add_test_helper(graph_IR, &addconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &addconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &addconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &addconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &addconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenAddScalarConvertsCorrectly) {
    // aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_scalar_graph("add", "2.2", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, true);
    add_test_helper(graph_IR, &addconverter, true, {3, 4, 3});
}

TEST(Converters, ATenAdd_TensorConvertsCorrectly) {
    // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_tensor_graph("add_", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, false);
    add_test_helper(graph_IR, &addconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &addconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenAdd_ScalarConvertsCorrectly) {
    // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("add_", "2.2", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, true);
    add_test_helper(graph_IR, &addconverter, true, {3, 4, 3});
}

TEST(Converters, ATenAddTensorAlphaConvertsCorrectly) {
    // aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add", "2.5");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, false);
    add_test_helper(graph_IR, &addconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &addconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &addconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &addconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &addconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenAddScalarAlphaConvertsCorrectly) {
    // aten::add.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("add", "2.2", "2.5");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, true);
    add_test_helper(graph_IR, &addconverter, true, {3, 4, 3});
}

TEST(Converters, ATenAdd_TensorAlphaConvertsCorrectly) {
    // aten::add_.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add_", "2.5");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, false);
    add_test_helper(graph_IR, &addconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &addconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenAdd_ScalarAlphaConvertsCorrectly) {
    // aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("add_", "2.2", "2.5");
    baidu::mirana::poros::AddConverter addconverter;
    add_test_helper(graph_IR, &addconverter, true);
    add_test_helper(graph_IR, &addconverter, true, {3, 4, 3});
}

TEST(Converters, ATenSubTensorConvertsCorrectly) {
    // aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("sub", "1.0");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, false);
    add_test_helper(graph_IR, &subconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &subconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &subconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &subconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &subconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenSubScalarConvertsCorrectly) {
    // aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_scalar_graph("sub", "2.2", "1.0");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, true);
    add_test_helper(graph_IR, &subconverter, true, {3, 4, 3});
}

TEST(Converters, ATenSub_TensorConvertsCorrectly) {
    // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_tensor_graph("sub_", "1.0");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, false);
    add_test_helper(graph_IR, &subconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &subconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenSub_ScalarConvertsCorrectly) {
    // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("sub_", "2.2", "1.0");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, true);
    add_test_helper(graph_IR, &subconverter, true, {3, 4, 3});
}

TEST(Converters, ATenSubTensorAlphaConvertsCorrectly) {
    // aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("sub", "2.5");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, false);
    add_test_helper(graph_IR, &subconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &subconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &subconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &subconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &subconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenSubScalarAlphaConvertsCorrectly) {
    // aten::sub.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("sub", "2.2", "2.5");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, true);
    add_test_helper(graph_IR, &subconverter, true, {3, 4, 3});
}

TEST(Converters, ATenSub_TensorAlphaConvertsCorrectly) {
    // aten::sub_.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    const auto graph_IR = gen_add_sub_tensor_graph("sub_", "2.5");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, false);
    add_test_helper(graph_IR, &subconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &subconverter, false, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenSub_ScalarAlphaConvertsCorrectly) {
    // aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
    const auto graph_IR = gen_add_sub_scalar_graph("sub_", "2.2", "2.5");
    baidu::mirana::poros::SubConverter subconverter;
    add_test_helper(graph_IR, &subconverter, true);
    add_test_helper(graph_IR, &subconverter, true, {3, 4, 3});
}

TEST(Converters, ATenRsubTensorConvertsCorrectly) {
    // aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = gen_add_sub_tensor_graph("rsub", "1.0");
    baidu::mirana::poros::RsubConverter rsubconverter;
    add_test_helper(graph_IR, &rsubconverter, false);
    add_test_helper(graph_IR, &rsubconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &rsubconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &rsubconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &rsubconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &rsubconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenRsubScalarConvertsCorrectly) {
    // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = gen_add_sub_scalar_graph("rsub", "2.2", "1.0");
    baidu::mirana::poros::RsubConverter rsubconverter;
    add_test_helper(graph_IR, &rsubconverter, true);
    add_test_helper(graph_IR, &rsubconverter, true, {3, 4, 3});
}

TEST(Converters, ATenRsubTensorAlphaConvertsCorrectly) {
    // aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = gen_add_sub_tensor_graph("rsub", "3.33");
    baidu::mirana::poros::RsubConverter rsubconverter;
    add_test_helper(graph_IR, &rsubconverter, false);
    add_test_helper(graph_IR, &rsubconverter, false, {3, 4}, {4});
    add_test_helper(graph_IR, &rsubconverter, false, {4}, {3, 4});
    add_test_helper(graph_IR, &rsubconverter, false, {4, 1}, {1, 4});
    add_test_helper(graph_IR, &rsubconverter, false, {3, 4, 3}, {4, 3});
    add_test_helper(graph_IR, &rsubconverter, false, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenRsubScalarAlphaConvertsCorrectly) {
    // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = gen_add_sub_scalar_graph("rsub", "2.2", "4.44");
    baidu::mirana::poros::RsubConverter rsubconverter;
    add_test_helper(graph_IR, &rsubconverter, true);
    add_test_helper(graph_IR, &rsubconverter, true, {3, 4, 3});
}

TEST(Converters, ATenRsubTensorTypePromotionConvertsCorrectly) {
    // aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : float = prim::Constant[value=3.33]()
          %3 : Tensor = aten::rsub(%0, %1, %2)
          return (%3))IR";
    baidu::mirana::poros::RsubConverter rsubconverter;

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({3,4,3}, {at::kCUDA}));
    input_data.push_back(at::ones({3,4,3}, {at::kCUDA}).to(at::ScalarType::Int));
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &rsubconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenRsubScalarTypePromotionConvertsCorrectly) {
    // aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=5]()
          %2 : float = prim::Constant[value=3.33]()
          %3 : Tensor = aten::rsub(%0, %1, %2)
          return (%3))IR";
    baidu::mirana::poros::RsubConverter rsubconverter;
    add_test_helper(graph_IR, &rsubconverter, true);
}

static void add_sub_dynamic_test_helper(const std::string& graph_IR, 
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

TEST(Converters, ATenAddIntdynamicConvertsCorrectly) {
    // aten::add.int(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int = aten::add(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::AddConverter addconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    add_sub_dynamic_test_helper(graph_IR, &addconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenSubIntdynamicConvertsCorrectly) {
    // aten::sub.int(int a, int b) -> (int)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int = aten::sub(%3, %4)
          %6 : Tensor = aten::add(%0, %5, %2)
          return (%6))IR";
    baidu::mirana::poros::SubConverter subconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    add_sub_dynamic_test_helper(graph_IR, &subconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenAddTdynamicConvertsCorrectly) {
    // aten::add.t(t[] a, t[] b) -> (t[])
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : int[] = aten::size(%0)
          %3 : int[] = aten::size(%1)
          %4 : int[] = aten::add(%2, %3)
          %5 : int = prim::Constant[value=2]()
          %6 : int = aten::__getitem__(%4, %5) 
          %7 : int = prim::Constant[value=1]()
          %8 : Tensor = aten::add(%0, %6, %7)
          return (%8))IR";
    baidu::mirana::poros::AddConverter addconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    input_data.push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[0].push_back(at::zeros({6, 7}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));

    add_sub_dynamic_test_helper(graph_IR, &addconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenAddTensordynamicConvertsCorrectly) {
    //dynamic tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({15, 1}, {at::kCUDA}));
    input_data.push_back(at::randn({300}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({40, 1}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({300}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({8, 1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({300}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({20, 1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({300}, {at::kCUDA}));

    add_sub_dynamic_test_helper(graph_IR, &addconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenAddTensordynamicMoreConvertsCorrectly) {
    //dynamic tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 1}, {at::kCUDA}));
    input_data.push_back(at::randn({300}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 1}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({400}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({100}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({200}, {at::kCUDA}));

    add_sub_dynamic_test_helper(graph_IR, &addconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenAddTensordynamicMore2ConvertsCorrectly) {
    //dynamic tensor
    const auto graph_IR = gen_add_sub_tensor_graph("add", "1.0");
    baidu::mirana::poros::AddConverter addconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 1, 45}, {at::kCUDA}));
    input_data.push_back(at::randn({300, 1}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({400, 1, 45}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({400, 1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 1, 45}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({100, 1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({100, 1, 45}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({200, 1}, {at::kCUDA}));

    add_sub_dynamic_test_helper(graph_IR, &addconverter, input_data, true, &prewarm_data);
}