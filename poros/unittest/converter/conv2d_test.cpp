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
* @file conv2d_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/util/test_util.h"
#include "poros/converter/gpu/convolution.h"

static void conv2d_test_helper(const std::string& graph_IR, 
                               baidu::mirana::poros::IConverter* converter,
                               std::vector<int64_t> shape_inputs,
                               std::vector<int64_t> shape_weights,
                               std::vector<int64_t> shape_bias) {
    std::vector<at::Tensor> input_data;
    // auto in = at::randn({1, 3, 10, 10}, {at::kCUDA});
    // auto w = at::randn({8, 3, 5, 5}, {at::kCUDA});
    // auto b = at::randn({8}, {at::kCUDA});
    auto in = at::randn(shape_inputs, {at::kCUDA});
    auto w = at::randn(shape_weights, {at::kCUDA});
    auto b = at::randn(shape_bias, {at::kCUDA});
    input_data.push_back(in);
    input_data.push_back(w);
    input_data.push_back(b);
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    //ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 0.0001));
}

TEST(Converters, ATenConv2dVggishTestConvertsCorrectly) {  
    // aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor, %2 : Tensor):
        %3 : int[] = prim::Constant[value=[1, 1]]()
        %4 : int[] = prim::Constant[value=[1, 1]]()
        %5 : int[] = prim::Constant[value=[1, 1]]()
        %6 : int = prim::Constant[value=1]()
        %7 : Tensor = aten::conv2d(%0, %1, %2, %3, %4, %5, %6)
        return (%7))IR";
    baidu::mirana::poros::ConvolutionConverter convolutionconverter;
    conv2d_test_helper(graph_IR, &convolutionconverter, {60, 256, 12, 8}, {512, 256, 3, 3}, {512});
}
