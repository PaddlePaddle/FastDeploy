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
* @file pooling_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/pooling.h"
#include "poros/util/test_util.h"

static void pooling_test_helper(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            std::vector<int64_t> shape) {
    std::vector<at::Tensor> input_data;
    // input_data.push_back(at::randn(shape, {at::kCUDA}));
    input_data.push_back(at::randint(-50, 50, shape, {at::kCUDA}));
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

static std::string gen_maxpool_graph(const std::string& op,
                                    const std::string& kernel_size,
                                    const std::string& stride,
                                    const std::string& padding,
                                    const std::string& dilation,
                                    const std::string& ceil_mode) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + kernel_size + R"IR(]]()
          %2 : int[] = prim::Constant[value=[)IR" + stride + R"IR(]]()
          %3 : int[] = prim::Constant[value=[)IR" + padding + R"IR(]]()
          %4 : int[] = prim::Constant[value=[)IR" + dilation + R"IR(]]()
          %5 : bool = prim::Constant[value=)IR" + ceil_mode + R"IR(]()
          %6 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2, %3, %4, %5)
          return (%6))IR";
}

static std::string gen_avgpool_graph(const std::string& op,
                                    const std::string& kernel_size,
                                    const std::string& stride,
                                    const std::string& padding,
                                    const std::string& ceil_mode,
                                    const std::string& count_include_pad,
                                    const std::string& divisor_override) {
    std::string divisor_ir("");
    std::string op_ir("");
    if (divisor_override.empty()) {
        divisor_ir = "None = prim::Constant()";
    } else {
        divisor_ir = "int = prim::Constant[value=" + divisor_override + "]()";
    }
    if (op == "avg_pool1d") {
        op_ir = op + "(%0, %1, %2, %3, %4, %5)";
    } else {
        op_ir = op + "(%0, %1, %2, %3, %4, %5, %6)";
    }
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + kernel_size + R"IR(]]()
          %2 : int[] = prim::Constant[value=[)IR" + stride + R"IR(]]()
          %3 : int[] = prim::Constant[value=[)IR" + padding + R"IR(]]()
          %4 : bool = prim::Constant[value=)IR" + ceil_mode + R"IR(]()
          %5 : bool = prim::Constant[value=)IR" + count_include_pad + R"IR(]()
          %6 : )IR" + divisor_ir + R"IR(
          %7 : Tensor = aten::)IR" + op_ir + R"IR(
          return (%7))IR";
}

TEST(Converters, ATenMaxPool1DConvertsCorrectly) {
    // aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool1d", "3", "2", "1", "1", "0");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8});
}

TEST(Converters, ATenMaxPool1DCeilConvertsCorrectly) { 
    // aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool1d", "3", "2", "1", "1", "1");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
}

TEST(Converters, ATenMaxPool2DConvertsCorrectly) {
    // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool2d", "3, 3", "2, 2", "1, 1", "1, 1", "0");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}

TEST(Converters, ATenMaxPool2DCeilConvertsCorrectly) { 
    // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool2d", "3, 3", "2, 2", "1, 1", "1, 1", "1");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
}

TEST(Converters, ATenMaxPool3DConvertsCorrectly) {
    // aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "1, 1, 1", "0");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}

TEST(Converters, ATenMaxPool3DCeilConvertsCorrectly) { 
    // aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
    const auto graph_IR = gen_maxpool_graph("max_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "1, 1, 1", "1");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
}

TEST(Converters, ATenAvgPool1DConvertsCorrectly) {
    // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool1d", "3", "2", "1", "0", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8});
}

TEST(Converters, ATenAvgPool1DCeilConvertsCorrectly) {
    // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool1d", "3", "2", "1", "1", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
    // pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8}); // fail
}

TEST(Converters, ATenAvgPool1DNoCountPadConvertsCorrectly) {
    // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool1d", "3", "2", "1", "0", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8});
}

TEST(Converters, ATenAvgPool1DCeilNoCountPadConvertsCorrectly) {
    // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool1d", "3", "2", "1", "1", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 1, 8});
}

TEST(Converters, ATenAvgPool2DConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "0", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}

TEST(Converters, ATenAvgPool2DCeilConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "1", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    // pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8}); // fail
}

TEST(Converters, ATenAvgPool2DNoCountPadConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "0", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}

TEST(Converters, ATenAvgPool2DCeilNoCountPadConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "1", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}

TEST(Converters, ATenAvgPool2DDivConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "0", "1", "4");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}

TEST(Converters, ATenAvgPool2DNegtiveDivConvertsCorrectly) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool2d", "3, 3", "2, 2", "1, 1", "0", "1", "-4");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 8, 8});
}


TEST(Converters, ATenAvgPool3DConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "0", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}

TEST(Converters, ATenAvgPool3DCeilConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "1", "1", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    // pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8}); // fail
}

TEST(Converters, ATenAvgPool3DNoCountPadConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "0", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}

TEST(Converters, ATenAvgPool3DCeilNoCountPadConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "1", "0", "");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}

TEST(Converters, ATenAvgPool3DDivConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "0", "1", "8");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}

TEST(Converters, ATenAvgPool3DNegtiveDivConvertsCorrectly) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    const auto graph_IR = gen_avgpool_graph("avg_pool3d", "3, 3, 3", "2, 2, 2", "1, 1, 1", "0", "1", "-8");
    baidu::mirana::poros::PoolingConverter poolingconverter;
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 7, 7, 7});
    pooling_test_helper(graph_IR, &poolingconverter, {1, 3, 8, 8, 8});
}