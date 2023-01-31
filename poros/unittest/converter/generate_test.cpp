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
* @file generate_test.cpp
* @author tianshaoqing@baidu.com
* @date Tue Nov 23 12:26:28 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/generate.h"
#include "poros/util/test_util.h"

static void generate_dy_test_helper(const std::string& graph_IR, 
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

TEST(Converters, ATenZeroslikeConvertsCorrectly) {
    // aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %zerosout : Tensor = aten::zeros_like(%0, %1, %1, %1, %1, %1)
          return (%zerosout))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    baidu::mirana::poros::ZerosLikeConverter zeroslikeconverter;
    generate_dy_test_helper(graph_IR, &zeroslikeconverter, input_data);
}

TEST(Converters, ATenZeroslikeDtypeConvertsCorrectly) {
    // aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    // scalar type index in aten and support situation ('o' is support and 'x' is not support):
    // uint8_t -> 0 x
    // int8_t -> 1 x
    // int16_t -> 2 x
    // int -> 3 o
    // int64_t -> 4 x
    // Half -> 5 o
    // float -> 6 o
    // bool -> 11 x
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %2 : int = prim::Constant[value=3]()
          %zerosout : Tensor = aten::zeros_like(%0, %2, %1, %1, %1, %1)
          return (%zerosout))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    baidu::mirana::poros::ZerosLikeConverter zeroslikeconverter;
    generate_dy_test_helper(graph_IR, &zeroslikeconverter, input_data);
}

TEST(Converters, ATenZeroslikeDynamicConvertsCorrectly) {
    // aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %zerosout : Tensor = aten::zeros_like(%0, %1, %1, %1, %1, %1)
          return (%zerosout))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::ZerosLikeConverter zeroslikeconverter;
    generate_dy_test_helper(graph_IR, &zeroslikeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenZeroslikeDynamicDtypeConvertsCorrectly) {
    // aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %2 : int = prim::Constant[value=5]()
          %zerosout : Tensor = aten::zeros_like(%0, %2, %1, %1, %1, %1)
          return (%zerosout))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::ZerosLikeConverter zeroslikeconverter;
    generate_dy_test_helper(graph_IR, &zeroslikeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenZerosDynamicConvertsCorrectly) {
    // aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : Tensor = aten::zeros(%1, %2, %2, %3, %2)
          return (%4))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::ZerosConverter ZerosConverter;
    generate_dy_test_helper(graph_IR, &ZerosConverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenZerosDynamicDtypeConvertsCorrectly) {
    // aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : int = prim::Constant[value=3]()
          %5 : Tensor = aten::zeros(%1, %4, %2, %3, %2)
          return (%5))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::ZerosConverter ZerosConverter;
    generate_dy_test_helper(graph_IR, &ZerosConverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenOnesDynamicConvertsCorrectly) {
    // aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : Tensor = aten::ones(%1, %2, %2, %3, %2)
          return (%4))IR";
          
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::OnesConverter onesconverter;
    generate_dy_test_helper(graph_IR, &onesconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenOnesDynamicDtypeConvertsCorrectly) {
    // aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : int = prim::Constant[value=5]()
          %5 : Tensor = aten::ones(%1, %4, %2, %3, %2)
          return (%5))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::OnesConverter onesconverter;
    generate_dy_test_helper(graph_IR, &onesconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenFullDynamicDtypeConvertsCorrectly) {
    // aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : int = prim::Constant[value=6]()
          %5 : Tensor = aten::full(%1, %4, %4, %2, %3, %2)
          return (%5))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::FullConverter fullconverter;
    generate_dy_test_helper(graph_IR, &fullconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenArangeDynamicDtypeConvertsCorrectly) {
    // aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=1]()
          %2 : int = aten::size(%0, %1)
          %3 : None = prim::Constant()
          %4 : Device = prim::Constant[value="cuda"]()
          %5 : int = prim::Constant[value=3]()
          %6 : Tensor = aten::arange(%2, %5, %3, %4, %3)
          return (%6))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    baidu::mirana::poros::ArangeConverter arangeconverter;
    generate_dy_test_helper(graph_IR, &arangeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenArangeStartEndDynamicDtypeConvertsCorrectly) {
    // aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %s.1 : int = aten::size(%0, %1)
          %s.2 : int = aten::size(%0, %2)
          %3 : None = prim::Constant()
          %4 : Device = prim::Constant[value="cuda"]()
          %5 : int = prim::Constant[value=3]()
          %6 : Tensor = aten::arange(%s.1, %s.2, %5, %3, %4, %3)
          return (%6))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({1, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({1, 2}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({1, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 5}, {at::kCUDA}));
    baidu::mirana::poros::ArangeConverter arangeconverter;
    generate_dy_test_helper(graph_IR, &arangeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenArangeStartConstantEndDynamicDtypeConvertsCorrectly) {
    // aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %s.1 : int = prim::Constant[value=-10]()
          %1 : int = prim::Constant[value=1]()
          %s.2 : int = aten::size(%0, %1)
          %3 : None = prim::Constant()
          %4 : Device = prim::Constant[value="cuda"]()
          %5 : int = prim::Constant[value=6]()
          %6 : Tensor = aten::arange(%s.1, %s.2, %5, %3, %4, %3)
          return (%6))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({1, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({1, 2}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({1, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 5}, {at::kCUDA}));
    baidu::mirana::poros::ArangeConverter arangeconverter;
    generate_dy_test_helper(graph_IR, &arangeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenTensorDynamicDtypeConvertsCorrectly) {
  const auto graph_IR = R"IR(
    graph(%0 : Tensor):
        %1 : bool = prim::Constant[value=0]()
        %2 : Device = prim::Constant[value="cuda:0"]()
        %3 : int = prim::Constant[value=6]()
        %4 : int[] = aten::size(%0)
        %5 : Tensor = aten::tensor(%4, %3, %2, %1)
        return (%5))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({11, 2, 1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({10, 2, 1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 2, 1}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 2, 1}, {at::kCUDA}));
    baidu::mirana::poros::TensorConverter tensorconverter;
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &tensorconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLinspaceScalarTensorConvertsCorrectly) {
    // aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
    // aten::linspace目前只能构造dynamic的单测，非dy的单测会被某些pass变为constant
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %2 : int = prim::Constant[value=0]()
            %3 : None = prim::Constant()
            %start : int = prim::Constant[value=-10]()
            %end : int = prim::Constant[value=100]()
            %step : int = aten::size(%0, %2)
            %device : Device = prim::Constant[value="cuda"]()
            %5 : Tensor = aten::linspace(%start, %end, %step, %3, %3, %device, %3)
            %6 : Tensor = aten::mul(%0, %5)
            return (%6))IR";
            
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::ones({6}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({6}, {at::kCUDA}));

    baidu::mirana::poros::LinspaceConverter linspaceconverter;
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &linspaceconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLinspaceStartEndDiffTypeConvertsCorrectly) {
    // aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %2 : int = prim::Constant[value=0]()
            %3 : None = prim::Constant()
            %start : int = prim::Constant[value=-10]()
            %end : float = prim::Constant[value=43.3]()
            %step : int = aten::size(%0, %2)
            %device : Device = prim::Constant[value="cuda"]()
            %5 : Tensor = aten::linspace(%start, %end, %step, %3, %3, %device, %3)
            %6 : Tensor = aten::mul(%0, %5)
            return (%6))IR";
            
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::ones({6}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({6}, {at::kCUDA}));

    baidu::mirana::poros::LinspaceConverter linspaceconverter;
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &linspaceconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLinspaceStepNoneConvertsCorrectly) {
    std::string graph_IR_str;
    if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 11) {
        // aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
        graph_IR_str = R"IR(
            graph(%0 : Tensor, %1 : Tensor):
                %2 : int = prim::Constant[value=0]()
                %3 : None = prim::Constant()
                %start : int = aten::size(%0, %2)
                %end : float = prim::Constant[value=43.3]()
                %device : Device = prim::Constant[value="cuda"]()
                %5 : Tensor = aten::linspace(%start, %end, %3, %3, %3, %device, %3)
                %6 : Tensor = aten::mul(%1, %5)
                return (%6))IR";
    } else {
        // aten::linspace(Scalar start, Scalar end, int steps, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
        graph_IR_str = R"IR(
            graph(%0 : Tensor, %1 : Tensor):
                %2 : int = prim::Constant[value=0]()
                %3 : None = prim::Constant()
                %start : int = aten::size(%0, %2)
                %end : float = prim::Constant[value=43.3]()
                %step : int = prim::Constant[value=100]()
                %device : Device = prim::Constant[value="cuda"]()
                %5 : Tensor = aten::linspace(%start, %end, %step, %3, %3, %device, %3)
                %6 : Tensor = aten::mul(%1, %5)
                return (%6))IR";
    }
    const std::string graph_IR = graph_IR_str;
            
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::ones({1}, {at::kCUDA}));
    input_data.push_back(at::ones({100}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::ones({6}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({100}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({1}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({100}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({1}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({100}, {at::kCUDA}));

    baidu::mirana::poros::LinspaceConverter linspaceconverter;
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &linspaceconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenFulllikeConvertsCorrectly) {
    // aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %scalar : float = prim::Constant[value=2.5]()
          %out : Tensor = aten::full_like(%0, %scalar, %1, %1, %1, %1, %1)
          return (%out))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    baidu::mirana::poros::FulllikeConverter fulllikeconverter;
    generate_dy_test_helper(graph_IR, &fulllikeconverter, input_data);
}

TEST(Converters, ATenFulllikeDefaultTypeConvertsCorrectly) {
    // aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %scalar : float = prim::Constant[value=2.5]()
          %out : Tensor = aten::full_like(%0, %scalar, %1, %1, %1, %1, %1)
          return (%out))IR";
    std::vector<at::Tensor> input_data;
    auto options_pyt_int = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    input_data.push_back(at::zeros({2, 3, 4}, options_pyt_int));
    baidu::mirana::poros::FulllikeConverter fulllikeconverter;
    generate_dy_test_helper(graph_IR, &fulllikeconverter, input_data);
}

TEST(Converters, ATenFulllikeDtypeConvertsCorrectly) {
    // aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
    // scalar type index in aten and support situation ('o' is support and 'x' is not support):
    // uint8_t -> 0 x
    // int8_t -> 1 x
    // int16_t -> 2 x
    // int -> 3 o
    // int64_t -> 4 x
    // Half -> 5 o
    // float -> 6 o
    // bool -> 11 x
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %1 : None = prim::Constant()
            %2 : int = prim::Constant[value=6]()
            %scalar : int = prim::Constant[value=2]()
            %out : Tensor = aten::full_like(%0, %scalar, %2, %1, %1, %1, %1)
            return (%out))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    baidu::mirana::poros::FulllikeConverter fulllikeconverter;
    generate_dy_test_helper(graph_IR, &fulllikeconverter, input_data);
}

TEST(Converters, ATenFulllikeDynamicConvertsCorrectly) {
    // aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %scalar : int = prim::Constant[value=2]()
          %out : Tensor = aten::full_like(%0, %scalar, %1, %1, %1, %1, %1)
          return (%out))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3, 4}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    baidu::mirana::poros::FulllikeConverter fulllikeconverter;
    generate_dy_test_helper(graph_IR, &fulllikeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenFulllikeDynamicDtypeConvertsCorrectly) {
    // aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : None = prim::Constant()
          %2 : int = prim::Constant[value=3]()
          %scalar : float = prim::Constant[value=2.5]()
          %out : Tensor = aten::full_like(%0, %scalar, %2, %1, %1, %1, %1)
          return (%out))IR";
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3, 4}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3, 4}, {at::kCUDA}));
    baidu::mirana::poros::FulllikeConverter fulllikeconverter;
    generate_dy_test_helper(graph_IR, &fulllikeconverter, input_data, true, &prewarm_data);
}