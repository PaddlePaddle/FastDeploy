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
* @file expand_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/expand.h"
#include "poros/util/test_util.h"

static void expand_test_helper(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            bool singleInput,
                            std::vector<int64_t> shape1 = {3, 1},
                            std::vector<int64_t> shape2 = {3, 1}){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    if (!singleInput){
        input_data.push_back(at::randn(shape2, {at::kCUDA}));
    }
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    // ASSERT_TRUE(baidu::mirana::poros::testutil::almostEqual(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_expand_graph(const std::string& size, const std::string& implicit) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + size + R"IR(]]()
          %2 : bool = prim::Constant[value=)IR" + implicit + R"IR(]()
          %3 : Tensor = aten::expand(%0, %1, %2)
          return (%3))IR";
}

static std::string gen_repeat_graph(const std::string& size) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + size + R"IR(]]()
          %2 : Tensor = aten::repeat(%0, %1)
          return (%2))IR";
}

TEST(Converters, ATenExpandSameDimConvertsCorrectly) {
    // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
    const auto graph_IR = gen_expand_graph("3, 4", "0");
    baidu::mirana::poros::ExpandConverter expandconverter;
    expand_test_helper(graph_IR, &expandconverter, true);
}

TEST(Converters, ATenExpandTileConvertsCorrectly) {
    // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
    // 若%2参数个数大于%1,则expand从后向前对齐
    // [3,1] [2,3,4] -> [2,3,4]
    // [3,1] [1,3,4] -> [1,3,4]
    // [3,1] [3,-1,4] -> [3,3,4]
    const auto graph_IR = gen_expand_graph("2, 3, 4", "0");          
    baidu::mirana::poros::ExpandConverter expandconverter;
    expand_test_helper(graph_IR, &expandconverter, true);
}

TEST(Converters, ATenExpandTileLastConvertsCorrectly) {
    // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
    const auto graph_IR = gen_expand_graph("1, 3, 4", "0");
    baidu::mirana::poros::ExpandConverter expandconverter;
    expand_test_helper(graph_IR, &expandconverter, true);
}

TEST(Converters, ATenExpandNegativeSizeConvertsCorrectly) { 
    // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
    // 1 means not changing the size of that dimension
    const auto graph_IR = gen_expand_graph("3, -1, 4", "0");
    baidu::mirana::poros::ExpandConverter expandconverter;
    expand_test_helper(graph_IR, &expandconverter, true);
}

TEST(Converters, ATenRepeatConvertsCorrectly) {
    // aten::repeat(Tensor self, int[] repeats) -> Tensor
    // output shape计算方法:参数向后对齐(如果%1与%2维度不同的话,同expand)，依次相乘
    // [3,1] [4,2] -> [12,2]
    // [2,3,2] [2,2,2] -> [4,6,4]
    // [3,1] [1,3,2] -> [1,9,2]
    const auto graph_IR = gen_repeat_graph("4, 2");
    baidu::mirana::poros::RepeatConverter repeatconverter;
    expand_test_helper(graph_IR, &repeatconverter, true);
}

TEST(Converters, ATenRepeat3dConvertsCorrectly) {
    // aten::repeat(Tensor self, int[] repeats) -> Tensor
    const auto graph_IR = gen_repeat_graph("2, 2, 2");
    baidu::mirana::poros::RepeatConverter repeatconverter;
    expand_test_helper(graph_IR, &repeatconverter, true, {2, 3, 2});
}

TEST(Converters, ATenRepeatExtraDimsConvertsCorrectly) {
    // aten::repeat(Tensor self, int[] repeats) -> Tensor
    const auto graph_IR = gen_repeat_graph("1, 3, 2");
    baidu::mirana::poros::RepeatConverter repeatconverter;
    expand_test_helper(graph_IR, &repeatconverter, true);
}

static void expand_dynamic_test_helper(const std::string& graph_IR, 
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
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenExpandFromSizedynamicConvertsCorrectly) {
    // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : int = prim::Constant[value=-1]()
          %3 : int[] = aten::size(%0)
          %B.1 : int, %H.1 : int, %W.1 : int, %C.1 : int = prim::ListUnpack(%3)
          %4 : int[] = prim::ListConstruct(%B.1, %2, %C.1)
          %5 : Tensor = aten::reshape(%0, %4)
          %6 : int[] = aten::size(%5) 
          %B.2 : int, %N.2 : int, %C.2 : int = prim::ListUnpack(%6)
          %7 : int[] = prim::ListConstruct(%B.2, %2, %2)
          %8 : bool = prim::Constant[value=0]()
          %9 : Tensor = aten::expand(%1, %7, %8) 
          return (%9))IR";
    baidu::mirana::poros::ExpandConverter expandconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 24, 24, 512}, {at::kCUDA}));
    input_data.push_back(at::randn({1, 1, 512}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 24, 24, 512}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 24, 24, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 24, 24, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({1, 1, 512}, {at::kCUDA}));

    expand_dynamic_test_helper(graph_IR, &expandconverter, input_data, true, &prewarm_data);
}


/*aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)*/
static std::string gen_expand_as_graph() {
    return R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %3 : Tensor = aten::expand_as(%0, %1)
          return (%3))IR";
}

TEST(Converters, ATenExpandAsConvertsCorrectly) {
    /*aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)*/
    const auto graph_IR = gen_expand_as_graph();
    baidu::mirana::poros::ExpandConverter expandconverter;

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    input_data.push_back(at::randn({2, 24, 1, 512}, {at::kCUDA}));

    expand_dynamic_test_helper(graph_IR, &expandconverter, input_data);
}

TEST(Converters, ATenExpandAsDynamicConvertsCorrectly) {
    /*aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)*/
    const auto graph_IR = gen_expand_as_graph();
    baidu::mirana::poros::ExpandConverter expandconverter;

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    input_data.push_back(at::randn({2, 24, 1, 512}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({4, 24, 1, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 24, 1, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({1, 1, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 24, 1, 512}, {at::kCUDA}));

    expand_dynamic_test_helper(graph_IR, &expandconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenExpandAsDynamicMoreConvertsCorrectly) {
    /*aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)*/
    const auto graph_IR = gen_expand_as_graph();
    baidu::mirana::poros::ExpandConverter expandconverter;

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({24, 1, 512}, {at::kCUDA}));
    input_data.push_back(at::randn({4, 24, 1, 512}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({24, 1, 512}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randn({4, 24, 1, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 1, 512}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 2, 1, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 1, 512}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 4, 1, 512}, {at::kCUDA}));

    expand_dynamic_test_helper(graph_IR, &expandconverter, input_data, true, &prewarm_data);
}