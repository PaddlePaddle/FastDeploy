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
* @file group_norm_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/group_norm.h"
#include "poros/util/test_util.h"

static void groupnorm_test_helper(const std::string& graph_IR,
                            std::vector<at::Tensor>& input_data) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::GroupNormConverter groupnormconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &groupnormconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenGroupNormConvertsCorrectly) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor,
            %gamma : Tensor,
            %beta : Tensor):
        %1: int = prim::Constant[value=2]()
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::group_norm(%0, %1, %gamma, %beta, %8, %7)
        return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    input_data.push_back(at::randn({10}, {at::kCUDA}));
    groupnorm_test_helper(graph_IR, input_data);
}

TEST(Converters, ATenGroupNormConvertsCorrectly2InputsGamma) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %gamma : Tensor):
        %1 : int = prim::Constant[value=20]()
        %2 : None = prim::Constant()
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::group_norm(%0, %1, %gamma, %2, %8, %7)
        return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 100, 50, 50}, {at::kCUDA}));
    input_data.push_back(at::randn({100}, {at::kCUDA}));
    groupnorm_test_helper(graph_IR, input_data);
}

TEST(Converters, ATenGroupNormConvertsCorrectlyOneInput) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=20]()
        %2 : None = prim::Constant()
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::group_norm(%0, %1, %2, %2, %8, %7)
        return (%9))IR";
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({4, 100, 50, 50}, {at::kCUDA}));
    groupnorm_test_helper(graph_IR, input_data);
}


static void groupnorm_dy_test_helper(const std::string& graph_IR, 
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    baidu::mirana::poros::GroupNormConverter groupnormconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &groupnormconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenGroupNormConvertsDynamicCorrectly) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor,
            %gamma : Tensor,
            %beta : Tensor):
        %1: int = prim::Constant[value=2]()
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::group_norm(%0, %1, %gamma, %beta, %8, %7)
        return (%9))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 3, 3}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({10}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({10}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));
    input_data.push_back(at::ones({10}, {at::kCUDA}));
    input_data.push_back(at::ones({10}, {at::kCUDA}));

    groupnorm_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, ATenGroupNormConvertsCorrectlyDynamic2Inputsgamma) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %gamma : Tensor):
          %1 : int = prim::Constant[value=2]()
          %2 : None = prim::Constant()
          %7 : bool = prim::Constant[value=0]()
          %8 : float = prim::Constant[value=1.0000000000000001e-05]()
          %9 : Tensor = aten::group_norm(%0, %1, %gamma, %2, %8, %7)
          return (%9))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 100, 50, 50}, {at::kCUDA}));
    prewarm_data[0].push_back(at::ones({100}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({10, 100, 40, 40}, {at::kCUDA}));
    prewarm_data[1].push_back(at::ones({100}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 100, 40, 40}, {at::kCUDA}));
    prewarm_data[2].push_back(at::ones({100}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 100, 40, 40}, {at::kCUDA}));
    input_data.push_back(at::ones({100}, {at::kCUDA}));

    groupnorm_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}

TEST(Converters, ATenGroupNormConvertsDynamicOneInputCorrectly) {
    // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
     const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=2]()
        %2 : None = prim::Constant()
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::group_norm(%0, %1, %2, %2, %8, %7)
        return (%9))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 6, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 10, 3, 3}, {at::kCUDA}));

    groupnorm_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}