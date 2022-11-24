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

#include "poros/converter/gpu/lstm_cell.h"
#include "poros/util/test_util.h"

static void linear_test_helper(const std::string& graph_IR,
                            const std::vector<at::Tensor>& input_data) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::LstmCellConverter lstm_cellconverter;

    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &lstm_cellconverter, 
                input_data, graph_output, poros_output));
    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenlstm_cellconverterCorrectly) {
    //aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
    
    const auto graph = R"IR(
        graph(%0 : Tensor,
          %1 : Tensor,
          %3 : Tensor,
          %4 : Tensor,
          %5 : Tensor,
          %6 : Tensor,
          %7 : Tensor):
          %2 : Tensor[] = prim::ListConstruct(%0, %1)
          %8 : Tensor, %9 : Tensor = aten::lstm_cell(%3, %2, %4, %5, %6, %7)
          return (%8))IR";

    std::vector<at::Tensor> input_data;
    auto input = at::randn({50, 10}, {at::kCUDA});
    auto h0 = at::randn({50, 20}, {at::kCUDA});
    auto c0 = at::randn({50, 20}, {at::kCUDA});
    auto w_ih = at::randn({4 * 20, 10}, {at::kCUDA});
    auto w_hh = at::randn({4 * 20, 20}, {at::kCUDA});
    auto b_ih = at::randn({4 * 20}, {at::kCUDA});
    auto b_hh = at::randn({4 * 20}, {at::kCUDA});
    
    input_data.push_back(h0);
    input_data.push_back(c0);
    input_data.push_back(input);
    input_data.push_back(w_ih);
    input_data.push_back(w_hh);
    input_data.push_back(b_ih);
    input_data.push_back(b_hh);

    linear_test_helper(graph, input_data);
}
