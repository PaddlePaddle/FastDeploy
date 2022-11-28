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
* @file lstm_cell_test.cpp
* @author wangrui39@baidu.com
* @date Mon December 13 11:36:11 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/lstm.h"
#include "poros/util/test_util.h"

static void lstm_test_helper(const std::string& graph_IR,
                            const std::vector<at::Tensor>& input_data,
                            baidu::mirana::poros::IConverter* converter) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU

    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter,
                input_data, graph_output, poros_output));
    ASSERT_EQ(3, graph_output.size());
    ASSERT_EQ(3, poros_output.size());

    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[1], poros_output[1], 2e-6));
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[2], poros_output[2], 2e-6));

}

TEST(Converters, ATenlstmconverterCorrectly) {
    // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
    // num_layers = 1
    // bidirectional = false
    // batch_first = false
    const auto graph = R"IR(
        graph( %0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor,
          %5 : Tensor,
          %6 : Tensor):
          %11 : bool = prim::Constant[value=1]()
          %12 : bool = prim::Constant[value=0]()
          %13 : int = prim::Constant[value=1]()
          %14 : float = prim::Constant[value=0.0]()
          %15 : Tensor[] = prim::ListConstruct(%0, %1)
          %16 : Tensor[] = prim::ListConstruct(%3, %4, %5, %6)
          %17 : Tensor, %18 : Tensor, %19 : Tensor = aten::lstm(%2, %15, %16, %11, %13, %14, %12, %12, %12)
          return (%17, %18, %19))IR";

    /*const auto graph = R"IR(
        graph( %0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor,
          %5 : Tensor,
          %6 : Tensor):
          %11 : bool = prim::Constant[value=1]()
          %12 : bool = prim::Constant[value=0]()
          %13 : int = prim::Constant[value=1]()
          %14 : float = prim::Constant[value=0.0]()
          %15 : Tensor[] = prim::ListConstruct(%0, %1)
          %16 : Tensor[] = prim::ListConstruct(%3, %4, %5, %6)
          %17 : Tensor, %18 : Tensor, %19 : Tensor = aten::lstm(%2, %15, %16, %11, %13, %14, %12, %12, %12)
          return (%17, %18, %19))IR";*/

    std::vector<at::Tensor> input_data;    
    auto input = at::randn({1, 5, 1}, {at::kCUDA});
    auto h0 = at::randn({1, 5, 2}, {at::kCUDA});
    auto c0 = at::randn({1, 5, 2}, {at::kCUDA});

    auto w1 = at::randn({8, 1}, {at::kCUDA});
    auto w2 = at::randn({8, 2}, {at::kCUDA});
    auto w3 = at::randn({8}, {at::kCUDA});
    auto w4 = at::randn({8}, {at::kCUDA});

    input_data.push_back(h0);
    input_data.push_back(c0);
    input_data.push_back(input);

    input_data.push_back(w1);
    input_data.push_back(w2);
    input_data.push_back(w3);
    input_data.push_back(w4);


    baidu::mirana::poros::LstmConverter lstmconverter;
    lstm_test_helper(graph, input_data, &lstmconverter);
}

TEST(Converters, ATenlstmconverterBidirectionalCorrectly) {
    // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
    // num_layers = 1
    // bidirectional = true
    // batch_first = true
    const auto graph = R"IR(
        graph( %0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor,
          %5 : Tensor,
          %6 : Tensor,
          %7 : Tensor,
          %8 : Tensor,
          %9 : Tensor,
          %10 : Tensor):
          %11 : bool = prim::Constant[value=1]()
          %12 : bool = prim::Constant[value=0]()
          %13 : int = prim::Constant[value=1]()
          %14 : float = prim::Constant[value=0.0]()
          %15 : Tensor[] = prim::ListConstruct(%0, %1)
          %16 : Tensor[] = prim::ListConstruct(%3, %4, %5, %6, %7, %8, %9, %10)
          %17 : Tensor, %18 : Tensor, %19 : Tensor = aten::lstm(%2, %15, %16, %11, %13, %14, %12, %11, %11)
          return (%17, %18, %19))IR";


    std::vector<at::Tensor> input_data;    
    auto input = at::randn({50, 7, 10}, {at::kCUDA});
    auto h0 = at::randn({2, 50, 20}, {at::kCUDA});
    auto c0 = at::randn({2, 50, 20}, {at::kCUDA});

    auto w1 = at::randn({80, 10}, {at::kCUDA});
    auto w2 = at::randn({80, 20}, {at::kCUDA});
    auto w3 = at::randn({80}, {at::kCUDA});
    auto w4 = at::randn({80}, {at::kCUDA});

    auto r_w1 = at::randn({80, 10}, {at::kCUDA});
    auto r_w2 = at::randn({80, 20}, {at::kCUDA});
    auto r_w3 = at::randn({80}, {at::kCUDA});
    auto r_w4 = at::randn({80}, {at::kCUDA});

    input_data.push_back(h0);
    input_data.push_back(c0);
    input_data.push_back(input);

    input_data.push_back(w1);
    input_data.push_back(w2);
    input_data.push_back(w3);
    input_data.push_back(w4);
    input_data.push_back(r_w1);
    input_data.push_back(r_w2);
    input_data.push_back(r_w3);
    input_data.push_back(r_w4);


    baidu::mirana::poros::LstmConverter lstmconverter;
    lstm_test_helper(graph, input_data, &lstmconverter);
}

TEST(Converters, ATenlstmconverterNumlayerCorrectly) {
    // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
    // num_layers > 1
    // bidirectional = false
    // batch_first = true
    const auto graph = R"IR(
        graph( %0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor,
          %5 : Tensor,
          %6 : Tensor,
          %7 : Tensor,
          %8 : Tensor,
          %9 : Tensor,
          %10 : Tensor):
          %11 : bool = prim::Constant[value=1]()
          %12 : bool = prim::Constant[value=0]()
          %13 : int = prim::Constant[value=2]()
          %14 : float = prim::Constant[value=0.0]()
          %15 : Tensor[] = prim::ListConstruct(%0, %1)
          %16 : Tensor[] = prim::ListConstruct(%3, %4, %5, %6, %7, %8, %9, %10)
          %17 : Tensor, %18 : Tensor, %19 : Tensor = aten::lstm(%2, %15, %16, %11, %13, %14, %12, %12, %11)
          return (%17, %18, %19))IR";

    std::vector<at::Tensor> input_data;    
    auto input = at::randn({50, 7, 10}, {at::kCUDA});
    auto h0 = at::randn({2, 50, 20}, {at::kCUDA});
    auto c0 = at::randn({2, 50, 20}, {at::kCUDA});

    auto num1_w1 = at::randn({80, 10}, {at::kCUDA});
    auto num1_w2 = at::randn({80, 20}, {at::kCUDA});
    auto num1_w3 = at::randn({80}, {at::kCUDA});
    auto num1_w4 = at::randn({80}, {at::kCUDA});
    auto num2_w1 = at::randn({80, 20}, {at::kCUDA});
    auto num2_w2 = at::randn({80, 20}, {at::kCUDA});
    auto num2_w3 = at::randn({80}, {at::kCUDA});
    auto num2_w4 = at::randn({80}, {at::kCUDA});

    input_data.push_back(h0);
    input_data.push_back(c0);
    input_data.push_back(input);

    input_data.push_back(num1_w1);
    input_data.push_back(num1_w2);
    input_data.push_back(num1_w3);
    input_data.push_back(num1_w4);
    input_data.push_back(num2_w1);
    input_data.push_back(num2_w2);
    input_data.push_back(num2_w3);
    input_data.push_back(num2_w4);

    baidu::mirana::poros::LstmConverter lstmconverter;
    lstm_test_helper(graph, input_data, &lstmconverter);
}
