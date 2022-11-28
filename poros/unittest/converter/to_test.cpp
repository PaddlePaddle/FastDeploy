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
* @file to_test.cpp
* @author wangrui39@baidu.com
* @date Sunday November 14 11:36:11 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/to.h"
#include "poros/util/test_util.h"

static void add_test_helper(const std::string& graph_IR, 
                            baidu::mirana::poros::IConverter* converter,
                            std::vector<int64_t> shape1 = {5},
                            std::vector<int64_t> shape2 = {5}){
    std::vector<at::Tensor> input_data;

    input_data.push_back(at::ones(shape1, {at::kCUDA}));
    input_data.push_back(at::ones(shape2, {at::kCUDA}));
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

static std::string gen_to_graph() {
    std::string graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : float = prim::Constant[value=2]()
        %3 : int = prim::Constant[value=3]()
        %4 : bool = prim::Constant[value=0]()
        %5 : None = prim::Constant()
        %6 : Tensor = aten::to(%0, %3, %4, %4, %5)
        %7 : Tensor = aten::to(%1, %6, %4, %4, %5)
        %35 : Device = prim::Constant[value="cuda"]()
        %6 : Tensor = aten::to(%6, %35, %3, %4, %4, %5)
        %7 : Tensor = aten::to(%7, %35, %3, %4, %4, %5)
        %8 : int = prim::Constant[value=1]()
        %9 : Tensor = aten::add(%6, %7, %8)
        return (%9))IR"; 
    return graph;
}

TEST(Converters, ATenToConvertsCorrectly) {
    const auto graph_IR = gen_to_graph();
    baidu::mirana::poros::ToConverter toconverter;
    add_test_helper(graph_IR, &toconverter, {3, 4}, {3, 4});
}