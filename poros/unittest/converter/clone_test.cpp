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
* @file clone_test.cpp
* @author tianshaoqing@baidu.com
* @date Tue Nov 23 12:26:28 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/clone.h"
#include "poros/util/test_util.h"

static void clone_dy_test_helper(const std::string& graph_IR, 
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    baidu::mirana::poros::CloneConverter cloneconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &cloneconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenCloneConvertsCorrectly) {
    // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %memory_format : None = prim::Constant[value=0]()
          %1 : Tensor = aten::clone(%0, %memory_format)
          %2 : Tensor = aten::relu(%1)
          return (%2))IR";
    
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 100, 100, 100}, {at::kCUDA}));
    
    clone_dy_test_helper(graph_IR, input_data);
}

TEST(Converters, ATenCloneConvertsDynamicCorrectly) {
    // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %memory_format : None = prim::Constant[value=0]()
          %1 : Tensor = aten::clone(%0, %memory_format)
          %2 : Tensor = aten::relu(%1)
          return (%2))IR";
    
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 150, 100, 100}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({10, 100, 50, 50}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 100, 50, 50}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({10, 100, 50, 50}, {at::kCUDA}));
    
    clone_dy_test_helper(graph_IR, input_data, true, &prewarm_data);
}