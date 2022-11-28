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
* @file layer_norm_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/layer_norm.h"
#include "poros/util/test_util.h"

static void layernorm_test_helper(const std::string& graph_IR,
                            std::vector<at::Tensor>& input_data) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::LayerNormConverter layernormconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &layernormconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectlyLast3Dims) {
    // aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor,
            %gamma : Tensor,
            %beta : Tensor):
        %1: int = prim::Constant[value=3]()
        %2: int = prim::Constant[value=100]()
        %3: int = prim::Constant[value=100]()
        %4 : int[] = prim::ListConstruct(%1, %2, %3)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 3, 100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({3, 100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({3, 100, 100}, {at::kCUDA}));
    layernorm_test_helper(graph_IR, input_data);
}

// 同conv2d
TEST(Converters, ATenLayerNormConvertsCorrectlyLast2Dims) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor,
              %gamma : Tensor,
              %beta : Tensor):
          %2: int = prim::Constant[value=100]()
          %3: int = prim::Constant[value=100]()
          %4 : int[] = prim::ListConstruct(%2, %3)
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
          return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 3, 100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({100, 100}, {at::kCUDA}));
    layernorm_test_helper(graph_IR, input_data);
}

TEST(Converters, ATenLayerNormConvertsCorrectlyLast1Dims) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor,
              %gamma : Tensor,
              %beta : Tensor):
          %3: int = prim::Constant[value=100]()
          %4 : int[] = prim::ListConstruct(%3)
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
          return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 3, 100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({100}, {at::kCUDA}));
    input_data.push_back(at::randn({100}, {at::kCUDA}));
    layernorm_test_helper(graph_IR, input_data);
}

TEST(Converters, ATenLayerNormConvertsCorrectly2InputsGamma) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor,
              %gamma: Tensor):
          %beta: None = prim::Constant()
          %1: int = prim::Constant[value=100]()
          %4 : int[] = prim::ListConstruct(%1)
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
          return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 3, 100, 100}, {at::kCUDA}));
    input_data.push_back(at::randn({100}, {at::kCUDA}));
    layernorm_test_helper(graph_IR, input_data);
}

static void layernorm_dy_test_helper(const std::string& graph_IR, 
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    baidu::mirana::poros::LayerNormConverter layernormconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &layernormconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectly3dDynamicInput1dNormalizedShape) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor,
              %gamma: Tensor,
              %beta: Tensor):
          %1: int = prim::Constant[value=4]()
          %4 : int[] = prim::ListConstruct(%1)
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
          return (%9))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({10, 3, 4}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({4}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({4}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    input_data.push_back(at::ones({4}, {at::kCUDA}));
    input_data.push_back(at::ones({4}, {at::kCUDA}));

    layernorm_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, ATenLayerNormConvertsCorrectly3dDynamicInput2dNormalizedShape) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor,
              %gamma : Tensor,
              %beta : Tensor):
          %2: int = prim::Constant[value=3]()
          %3: int = prim::Constant[value=4]()
          %4 : int[] = prim::ListConstruct(%2, %3)
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
          return (%9))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({10, 3, 4}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({3, 4}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({3, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({3, 4}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({3, 4}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({3, 4}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({5, 3, 4}, {at::kCUDA}));
    input_data.push_back(at::ones({3, 4}, {at::kCUDA}));
    input_data.push_back(at::ones({3, 4}, {at::kCUDA}));

    layernorm_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}