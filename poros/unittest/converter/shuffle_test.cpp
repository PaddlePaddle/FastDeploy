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
* @file shuffle_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/util/test_util.h"
#include "poros/converter/gpu/shuffle.h"

static void shuffle_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                std::vector<int64_t> shape){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static void shuffle_dy_test_helper(const std::string& graph_IR, 
                                const std::vector<at::Tensor>& input_data,
                                baidu::mirana::poros::IConverter* converter,
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
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

std::string gen_double_int_graph(const std::string& op, 
                                const std::string& first_int,
                                const std::string& second_int) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + first_int + R"IR(]()
          %2 : int = prim::Constant[value=)IR" + second_int + R"IR(]()
          %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
          return (%3))IR";
}

std::string gen_int_list_graph(const std::string& op, const std::string& int_list) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + int_list + R"IR(]]()
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          return (%2))IR";
}

std::string gen_pixel_shuffle_graph(const std::string& upscale_factor) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + upscale_factor + R"IR(]()
          %2 : Tensor = aten::pixel_shuffle(%0, %1)
          return (%2))IR";
}

TEST(Converters, ATenTransposeConvertsCorrectly) {
    // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    const auto graph_IR = gen_double_int_graph("transpose", "1", "2");
    baidu::mirana::poros::TransposeConverter transposeconverter;
    shuffle_test_helper(graph_IR, &transposeconverter, {2, 3, 4});
}

TEST(Converters, ATenTransposeNegaiveConvertsCorrectly) {
    // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    const auto graph_IR = gen_double_int_graph("transpose", "-1", "-3");
    baidu::mirana::poros::TransposeConverter transposeconverter;
    shuffle_test_helper(graph_IR, &transposeconverter, {2, 3, 4, 5, 6});
}

TEST(Converters, ATenViewConvertsCorrectly) {
    // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
    const auto graph_IR = gen_int_list_graph("view", "1, 6");
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    shuffle_test_helper(graph_IR, &permuteviewconverter, {2, 3});
}

TEST(Converters, ATenViewNegtiveConvertsCorrectly) {
    // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
    const auto graph_IR = gen_int_list_graph("view", "-1, 8");
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    shuffle_test_helper(graph_IR, &permuteviewconverter, {4, 4});
}

TEST(Converters, ATenPermuteConvertsCorrectly) {
    // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
    const auto graph_IR = gen_int_list_graph("permute", "1, 0");
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    shuffle_test_helper(graph_IR, &permuteviewconverter, {2, 3});
}

TEST(Converters, ATenPermute3DConvertsCorrectly) {
    // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
    const auto graph_IR = gen_int_list_graph("permute", "1, 2, 0");
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    shuffle_test_helper(graph_IR, &permuteviewconverter, {1, 2, 3});
}

TEST(Converters, ATenPermute5DConvertsCorrectly) {
    // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
    const auto graph_IR = gen_int_list_graph("permute", "3, 1, 0, 2, 4");
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    shuffle_test_helper(graph_IR, &permuteviewconverter, {2, 3, 4, 5, 1});
}

TEST(Converters, ATenReshapeConvertsCorrectly) {
  // aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
  const auto graph_IR = gen_int_list_graph("reshape", "3, 2");
  baidu::mirana::poros::ReshapeConverter reshapeconverter;
  shuffle_test_helper(graph_IR, &reshapeconverter, {2, 3});
}

TEST(Converters, ATenReshapeNegtiveConvertsCorrectly) {
  // aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
  const auto graph_IR = gen_int_list_graph("reshape", "-1, 8");
  baidu::mirana::poros::ReshapeConverter reshapeconverter;
  shuffle_test_helper(graph_IR, &reshapeconverter, {4, 4});
}

TEST(Converters, ATenFlattenConvertsCorrectly) {
  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  const auto graph_IR = gen_double_int_graph("flatten", "0", "-1");
  baidu::mirana::poros::FlattenConverter flattenconverter;
  shuffle_test_helper(graph_IR, &flattenconverter, {1, 2, 3});
}

TEST(Converters, ATenFlattenStartEnddimConvertsCorrectly) {
  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  const auto graph_IR = gen_double_int_graph("flatten", "1", "2");
  baidu::mirana::poros::FlattenConverter flattenconverter;
  shuffle_test_helper(graph_IR, &flattenconverter, {1, 2, 3});
}

TEST(Converters, ATenT1DConvertsCorrectly) {
  // aten::t(Tensor(a) self) -> Tensor(a)
  const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : Tensor = aten::t(%0)
          %2 : Tensor = aten::relu(%1)
          return (%2))IR";
  baidu::mirana::poros::AtenTConverter atentConverter;
  shuffle_test_helper(graph_IR, &atentConverter, {5});
}

TEST(Converters, ATenT2DConvertsCorrectly) {
  // aten::t(Tensor(a) self) -> Tensor(a)
  const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : Tensor = aten::t(%0)
          return (%1))IR";
  baidu::mirana::poros::AtenTConverter atentConverter;
  shuffle_test_helper(graph_IR, &atentConverter, {5, 6});
}

TEST(Converters, ATenPixelShuffleConvertsCorrectly) {
  // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
  const auto graph_IR = gen_pixel_shuffle_graph("3");
  baidu::mirana::poros::PixelShuffleConverter pixelshuffleconverter;
  shuffle_test_helper(graph_IR, &pixelshuffleconverter, {1, 9, 4, 4});
}

TEST(Converters, ATenPixelShuffle3DConvertsCorrectly) {
  // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
  const auto graph_IR = gen_pixel_shuffle_graph("3");
  baidu::mirana::poros::PixelShuffleConverter pixelshuffleconverter;
  shuffle_test_helper(graph_IR, &pixelshuffleconverter, {9, 5, 6});
}

TEST(Converters, ATenPixelShuffle5DConvertsCorrectly) {
  // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
  const auto graph_IR = gen_pixel_shuffle_graph("3");
  baidu::mirana::poros::PixelShuffleConverter pixelshuffleconverter;
  shuffle_test_helper(graph_IR, &pixelshuffleconverter, {7, 8, 9, 5, 6});
}

static void shuffle_dynamic_test_helper(const std::string& graph_IR, 
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

TEST(Converters, ATenViewdynamicConvertsCorrectly) {
    // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::size(%0, %1)
          %4 : int = aten::size(%0, %2)
          %5 : int[] = prim::ListConstruct(%4, %3)
          %6 : Tensor = aten::view(%0, %5)
          return (%6))IR";
    baidu::mirana::poros::PermuteViewConverter permuteviewconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3}, {at::kCUDA}));

    shuffle_dynamic_test_helper(graph_IR, &permuteviewconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenReshapedynamicConvertsCorrectly) {
    // aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : int, %3 : int = prim::ListUnpack(%1)
          %4 : int[] = prim::ListConstruct(%3, %2)
          %5 : Tensor = aten::reshape(%0, %4)
          return (%5))IR";
    baidu::mirana::poros::ReshapeConverter reshapeconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 3}, {at::kCUDA}));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({2, 3}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({2, 3}, {at::kCUDA}));

    shuffle_dynamic_test_helper(graph_IR, &reshapeconverter, input_data, true, &prewarm_data);
}

TEST(Converters, ATenFlattenConvertsDynamicCorrectly) {
  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  const auto graph_IR = gen_double_int_graph("flatten", "0", "2");
  
  std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
  prewarm_data[0].push_back(at::randn({10, 64, 128}, {at::kCUDA}));
  prewarm_data[1].push_back(at::randn({5, 32, 64}, {at::kCUDA}));
  prewarm_data[2].push_back(at::randn({5, 32, 64}, {at::kCUDA}));

  std::vector<at::Tensor> input_data;
  input_data.push_back(at::randn({5, 32, 64}, {at::kCUDA}));
  baidu::mirana::poros::FlattenConverter flattenconverter;
  shuffle_dy_test_helper(graph_IR, input_data, &flattenconverter, true, &prewarm_data);
}

TEST(Converters, ATenFlattenConvertsDynamicNegStartEndCorrectly) {
  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  const auto graph_IR = gen_double_int_graph("flatten", "-3", "-2");
  
  std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
  prewarm_data[0].push_back(at::randn({10, 64, 128, 32}, {at::kCUDA}));
  prewarm_data[1].push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));
  prewarm_data[2].push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));

  std::vector<at::Tensor> input_data;
  input_data.push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));
  baidu::mirana::poros::FlattenConverter flattenconverter;
  shuffle_dy_test_helper(graph_IR, input_data, &flattenconverter, true, &prewarm_data);
}

TEST(Converters, ATenFlattenConvertsDynamicStartEqualEndCorrectly) {
  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
  const auto graph_IR = gen_double_int_graph("flatten", "1", "1");
  
  std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
  prewarm_data[0].push_back(at::randn({10, 64, 128, 32}, {at::kCUDA}));
  prewarm_data[1].push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));
  prewarm_data[2].push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));

  std::vector<at::Tensor> input_data;
  input_data.push_back(at::randn({5, 32, 64, 16}, {at::kCUDA}));
  baidu::mirana::poros::FlattenConverter flattenconverter;
  shuffle_dy_test_helper(graph_IR, input_data, &flattenconverter, true, &prewarm_data);
}