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
* @file shape_handle_test.cpp
* @author tianshaoqing@baidu.com
* @date Tues Jul 27 14:24:21 CST 2022
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/shape_handle.h"
#include "poros/util/test_util.h"

static void shape_handle_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                std::vector<int64_t> shape,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));

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

TEST(Converters, ATenShapeAsTensorConvertsCorrectly) {
    // aten::_shape_as_tensor(Tensor self) -> (Tensor)
    // aten::_shape_as_tensor output tensor is default on cpu. 
    // To keep all data on same device, need to add aten::to.device.
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %1 : Tensor = aten::_shape_as_tensor(%0)
            %2 : Device = prim::Constant[value="cuda"]()
            %3 : int = prim::Constant[value=3]()
            %4 : bool = prim::Constant[value=0]()
            %5 : None = prim::Constant()
            %6 : Tensor = aten::to(%1, %2, %3, %4, %4, %5)
            return (%6))IR";
    baidu::mirana::poros::ShapeastensorConverter shapeastensorconverter;
    shape_handle_test_helper(graph_IR, &shapeastensorconverter, {4, 5, 3, 1});
}

TEST(Converters, ATenShapeAsTensorDynamicConvertsCorrectly) {
    // aten::_shape_as_tensor(Tensor self) -> (Tensor)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %1 : Tensor = aten::_shape_as_tensor(%0)
            %2 : Device = prim::Constant[value="cuda"]()
            %3 : int = prim::Constant[value=3]()
            %4 : bool = prim::Constant[value=0]()
            %5 : None = prim::Constant()
            %6 : Tensor = aten::to(%1, %2, %3, %4, %4, %5)
            return (%6))IR";
    baidu::mirana::poros::ShapeastensorConverter shapeastensorconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    shape_handle_test_helper(graph_IR, &shapeastensorconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

// aten::len.Tensor(Tensor t) -> (int)
// aten::len.t(t[] a) -> (int)
TEST(Converters, ATenLenDynamicConvertsCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
            %1 : int = aten::len(%0)
            %2 : NoneType = prim::Constant()
            %3 : bool = prim::Constant[value=0]()
            %4 : Device = prim::Constant[value="cuda:0"]()
            %5 : Tensor = aten::tensor(%1, %2, %4, %3)
            return (%5))IR";

    baidu::mirana::poros::LenConverter lenconverter;

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
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &lenconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}