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
* @file reduce_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/reduce.h"
#include "poros/util/test_util.h"

static void reduce_test_helper(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            std::vector<int64_t> shape1,
                            bool single_input = true,
                            std::vector<int64_t> shape2 = {4, 4},
                            bool single_output = true,
                            bool int_flag = false){
    std::vector<at::Tensor> input_data;

    if(int_flag) {
        auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
        input_data.push_back(at::randint(1000, shape1, options_pyt_long));
    } else {
        input_data.push_back(at::randn(shape1, {at::kCUDA}));
    }

    if (!single_input){
        input_data.push_back(at::randn(shape2, {at::kCUDA}));
    }
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    if (single_output) {
        ASSERT_EQ(1, graph_output.size());
        ASSERT_EQ(1, poros_output.size());
    } else {
        ASSERT_EQ(2, graph_output.size());
        ASSERT_EQ(2, poros_output.size());
    }

    for (size_t i = 0; i < graph_output.size(); i++) {
        ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[i], poros_output[i], 2e-6));
    }
}

static std::string gen_basic_graph(const std::string& op) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : None = prim::Constant()
        %2 : Tensor = aten::)IR" +
        op + R"IR((%0, %1)
        return (%2))IR";
}

static std::string gen_min_max_graph(const std::string& op) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : Tensor = aten::)IR" +
        op + R"IR((%0)
        return (%1))IR";
}

static std::string gen_min_max_other_graph(const std::string& op) {
    return R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %1 : Tensor = aten::)IR" +
        op + R"IR((%0, %1)
        return (%1))IR";
}

static std::string gen_min_max_dim_graph(const std::string& op, const std::string& dim) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
        %2 : bool = prim::Constant[value=0]()
        %3 : Tensor, %4 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
        return (%3, %4))IR";
}

static std::string gen_argmin_argmax_graph(const std::string& op, const std::string& dim, const std::string& keepdim) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
        %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
        %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
        return (%3))IR";
}

static std::string gen_argmin_argmax_dim_none_graph(const std::string& op, const std::string& keepdim) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : None = prim::Constant()
        %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
        %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
        return (%3))IR";
}

static std::string gen_mean_sum_dim_graph(const std::string& op, const std::string& dim, const std::string& keepdim) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int[] = prim::Constant[value=[)IR" + dim + R"IR(]]()
          %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
          %3 : None = prim::Constant()
          %4 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2, %3)
          return (%4))IR";
}

static std::string gen_prod_dim_graph(const std::string& op, const std::string& dim, const std::string& keepdim) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
          %3 : None = prim::Constant()
          %4 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2, %3)
          return (%4))IR";
}

TEST(Converters, ATenMeanConvertsCorrectly) {
    // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_basic_graph("mean");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4});
}

TEST(Converters, ATenMeanDimConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "1", "0");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenMeanMltiDimsConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "0, 1", "0");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenMeanKeepDimsConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "1", "1");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4});
}

TEST(Converters, ATenMeanDimNegOneIndexConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "-1", "0");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenMeanDimNegOneIndexKeepDimsConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "-1", "1");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenMeanDimNegIndexConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "-2", "0");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenMeanDimNegIndexKeepDimsConvertsCorrectly) {
    // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("mean", "-2", "1");
    baidu::mirana::poros::MeanConverter meanconverter;
    reduce_test_helper(graph_IR, &meanconverter, {4, 4, 4});
}

TEST(Converters, ATenSumConvertsCorrectly) {
    // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_basic_graph("sum");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4});
}

TEST(Converters, ATenSumDimConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "1", "0");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenSumMltiDimsConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "0, 1", "0");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenSumKeepDimsConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "1", "1");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4});
}

TEST(Converters, ATenSumDimNegOneIndexConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "-1", "0");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenSumDimNegOneIndexKeepDimsConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "-1", "1");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenSumDimNegIndexConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "-2", "0");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenSumDimNegIndexKeepDimsConvertsCorrectly) {
    // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_mean_sum_dim_graph("sum", "-2", "1");
    baidu::mirana::poros::SumConverter sumconverter;
    reduce_test_helper(graph_IR, &sumconverter, {4, 4, 4});
}

TEST(Converters, ATenProdConvertsCorrectly) {
    // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_basic_graph("prod");
    baidu::mirana::poros::ProdConverter prodconverter;
    reduce_test_helper(graph_IR, &prodconverter, {4, 4});
}

TEST(Converters, ATenProdDimConvertsCorrectly) {
    // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_prod_dim_graph("prod", "1", "0");
    baidu::mirana::poros::ProdConverter prodconverter;
    reduce_test_helper(graph_IR, &prodconverter, {4, 4, 4});
}

TEST(Converters, ATenProdKeepDimsConvertsCorrectly) {
    // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    const auto graph_IR = gen_prod_dim_graph("prod", "1", "1");
    baidu::mirana::poros::ProdConverter prodconverter;
    reduce_test_helper(graph_IR, &prodconverter, {4, 4});
}

TEST(Converters, ATenMaxConvertsCorrectly) {
    // aten::max(Tensor self) -> Tensor
    const auto graph_IR = gen_min_max_graph("max");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 4});
}

TEST(Converters, ATenMinConvertsCorrectly) {
    // aten::min(Tensor self) -> Tensor
    const auto graph_IR = gen_min_max_graph("min");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 4});
}

TEST(Converters, ATenMaxOtherConvertsCorrectly) {
    // aten::max.other(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_min_max_other_graph("max");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 4}, false, {4, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {3, 4}, false, {4});
    reduce_test_helper(graph_IR, &maxminconverter, {4}, false, {3, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {4, 1}, false, {1, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {3, 4, 3}, false, {4, 3});
    reduce_test_helper(graph_IR, &maxminconverter, {4, 3}, false, {3, 4, 3});
}

TEST(Converters, ATenMinOtherConvertsCorrectly) {
    // aten::min.other(Tensor self, Tensor other) -> Tensor
    const auto graph_IR = gen_min_max_other_graph("min");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 4}, false, {4, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {3, 4}, false, {4});
    reduce_test_helper(graph_IR, &maxminconverter, {4}, false, {3, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {4, 1}, false, {1, 4});
    reduce_test_helper(graph_IR, &maxminconverter, {3, 4, 3}, false, {4, 3});
    reduce_test_helper(graph_IR, &maxminconverter, {4, 3}, false, {3, 4, 3});
}

TEST(Converters, ATenMaxDimConvertsCorrectly) {
    // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_min_max_dim_graph("max", "0");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR2 = gen_min_max_dim_graph("max", "1");
    reduce_test_helper(graph_IR2, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR3 = gen_min_max_dim_graph("max", "-1");
    reduce_test_helper(graph_IR3, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR4 = gen_min_max_dim_graph("max", "-1");
    reduce_test_helper(graph_IR4, &maxminconverter, {4, 3}, true, {}, false);
}

TEST(Converters, ATenMinDimConvertsCorrectly) {
    // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_min_max_dim_graph("min", "0");
    baidu::mirana::poros::MaxMinConverter maxminconverter;
    reduce_test_helper(graph_IR, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR2 = gen_min_max_dim_graph("min", "1");
    reduce_test_helper(graph_IR2, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR3 = gen_min_max_dim_graph("min", "-1");
    reduce_test_helper(graph_IR3, &maxminconverter, {4, 5, 3}, true, {}, false);
    const auto graph_IR4 = gen_min_max_dim_graph("min", "-1");
    reduce_test_helper(graph_IR4, &maxminconverter, {4, 3}, true, {}, false);
}

TEST(Converters, ATenMaxDimDynamicConvertsCorrectly) {
    // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_min_max_dim_graph("max", "0");
    baidu::mirana::poros::MaxMinConverter maxminconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({3, 4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({3, 4, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({3, 4, 5}, {at::kCUDA}));
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maxminconverter, 
                input_data, graph_output, poros_output, &prewarm_data));
    ASSERT_EQ(2, graph_output.size());
    ASSERT_EQ(2, poros_output.size());
    
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[1], poros_output[1], 2e-6));
}

TEST(Converters, ATenMinDimDynamicConvertsCorrectly) {
    // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    const auto graph_IR = gen_min_max_dim_graph("min", "1");
    baidu::mirana::poros::MaxMinConverter maxminconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({3, 4, 5}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({3, 4, 5}, {at::kCUDA}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({3, 4, 5}, {at::kCUDA}));
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maxminconverter, 
                input_data, graph_output, poros_output, &prewarm_data));
    ASSERT_EQ(2, graph_output.size());
    ASSERT_EQ(2, poros_output.size());
    
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[1], poros_output[1], 2e-6));
}

TEST(Converters, ArgmaxConvertersCorrectly) {
    // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
    baidu::mirana::poros::ArgmaxArgminConverter argmaxargminconverter;

    const auto graph_IR1 = gen_argmin_argmax_graph("argmax", "0", "0");
    reduce_test_helper(graph_IR1, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR2 = gen_argmin_argmax_graph("argmax", "1", "0");
    reduce_test_helper(graph_IR2, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR3 = gen_argmin_argmax_graph("argmax", "2", "0");
    reduce_test_helper(graph_IR3, &argmaxargminconverter, {4, 4, 6}, true, {}, true);
    const auto graph_IR4 = gen_argmin_argmax_graph("argmax", "3", "0");
    reduce_test_helper(graph_IR4, &argmaxargminconverter, {4, 4, 6, 8}, true, {}, true);

    const auto graph_IR5 = gen_argmin_argmax_graph("argmax", "0", "1");
    reduce_test_helper(graph_IR5, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR6 = gen_argmin_argmax_graph("argmax", "1", "1");
    reduce_test_helper(graph_IR6, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR7 = gen_argmin_argmax_graph("argmax", "-1", "1");
    reduce_test_helper(graph_IR7, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR8 = gen_argmin_argmax_graph("argmax", "-1", "0");
    reduce_test_helper(graph_IR8, &argmaxargminconverter, {4, 4}, true, {}, true);

    // test input tensor of int type
    const auto graph_IR9 = gen_argmin_argmax_graph("argmax", "1", "0");
    reduce_test_helper(graph_IR9, &argmaxargminconverter, {4, 4}, true, {}, true, true);
    const auto graph_IR10 = gen_argmin_argmax_graph("argmax", "-1", "0");
    reduce_test_helper(graph_IR10, &argmaxargminconverter, {4, 4}, true, {}, true, true);
}

TEST(Converters, ArgminConvertersCorrectly) {
    // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
    baidu::mirana::poros::ArgmaxArgminConverter argmaxargminconverter;

    const auto graph_IR1 = gen_argmin_argmax_graph("argmin", "0", "0");
    reduce_test_helper(graph_IR1, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR2 = gen_argmin_argmax_graph("argmin", "1", "0");
    reduce_test_helper(graph_IR2, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR3 = gen_argmin_argmax_graph("argmin", "2", "0");
    reduce_test_helper(graph_IR3, &argmaxargminconverter, {4, 4, 6}, true, {}, true);
    const auto graph_IR4 = gen_argmin_argmax_graph("argmin", "3", "0");
    reduce_test_helper(graph_IR4, &argmaxargminconverter, {4, 4, 6, 8}, true, {}, true);

    const auto graph_IR5 = gen_argmin_argmax_graph("argmin", "0", "1");
    reduce_test_helper(graph_IR5, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR6 = gen_argmin_argmax_graph("argmin", "1", "1");
    reduce_test_helper(graph_IR6, &argmaxargminconverter, {4, 4}, true, {}, true);
    const auto graph_IR7 = gen_argmin_argmax_graph("argmin", "-1", "1");
    reduce_test_helper(graph_IR7, &argmaxargminconverter, {4, 4}, true, {}, true);

    // test input tensor of int type
    const auto graph_IR9 = gen_argmin_argmax_graph("argmin", "1", "0");
    reduce_test_helper(graph_IR9, &argmaxargminconverter, {4, 4}, true, {}, true, true);
    const auto graph_IR10 = gen_argmin_argmax_graph("argmin", "-1", "0");
    reduce_test_helper(graph_IR10, &argmaxargminconverter, {4, 4}, true, {}, true, true);
}

// TODO: to imp dim=None
// TEST(Converters, ArgmaxNoneDimConvertersCorrectly) {
//     // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
//     baidu::mirana::poros::ArgmaxArgminConverter argmaxargminconverter;
//     const auto graph_IR1 = gen_argmin_argmax_dim_none_graph("argmax", "0");
//     reduce_test_helper(graph_IR1, &argmaxargminconverter, {4, 4}, true, {}, true);
// }