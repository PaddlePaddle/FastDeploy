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
* @file roll_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Jul 20 19:34:51 CST 2022
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/roll.h"
#include "poros/util/test_util.h"

static void roll_test_helper(const std::string& graph_IR, 
                                std::vector<int64_t> shape,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    std::vector<at::Tensor> input_data;
    int64_t shape_mul = 1;
    for (int64_t& s : shape) {
        shape_mul *= s;
    }
    input_data.push_back(at::randint(0, shape_mul, shape, {at::kCUDA}));

    baidu::mirana::poros::RollConverter rollconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &rollconverter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_roll_graph(const std::string& shifts, const std::string& dims) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=)IR" + shifts + R"IR(]()
          %2 : int[] = prim::Constant[value=)IR" + dims + R"IR(]()
          %3 : Tensor = aten::roll(%0, %1, %2)
          return (%3))IR";
}

TEST(Converters, ATenRollConvertsCorrectly) {
    // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)
    const std::string graph_IR = gen_roll_graph("[-1, 0, -2, 3]", "[0, 1, 2, 3]");
    roll_test_helper(graph_IR, {4, 4, 4, 4});
}


TEST(Converters, ATenRollConvertsCorrectlyShiftsGreaterThanDims) {
    // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)
    const std::string graph_IR = gen_roll_graph("[-99, 100, 51, -21]", "[0, 1, 2, 3]");
    roll_test_helper(graph_IR, {4, 4, 4, 4});
}

TEST(Converters, ATenRollConvertsCorrectlyShiftSomeDims) {
    // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)
    const std::string graph_IR = gen_roll_graph("[0, -2, 3]", "[0, 1, 3]");
    roll_test_helper(graph_IR, {4, 4, 4, 4});
}