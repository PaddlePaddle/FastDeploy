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
* @file batch_norm_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/batch_norm.h"
#include "poros/util/test_util.h"

TEST(Converters, ATenBatchnormalConvertsCorrectly) {
    // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1: Tensor, %2: Tensor, %3: Tensor, %4: Tensor):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

        auto in = at::randn({1, 5, 5, 5}, {at::kCUDA});
        auto gamma = at::randn({5}, {at::kCUDA});
        auto beta = at::randn({5}, {at::kCUDA});
        auto mean = at::randn({5}, {at::kCUDA});
        auto var = at::randn({5}, {at::kCUDA}).abs();

        baidu::mirana::poros::PorosOptions poros_option; // default device GPU
        baidu::mirana::poros::BatchNormConverter batchnormconverter;

        // 运行原图与engine获取结果
        std::vector<at::Tensor> graph_output;
        std::vector<at::Tensor> poros_output;
        ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &batchnormconverter, 
                    {in, gamma, beta, mean, var}, graph_output, poros_output));

        ASSERT_EQ(1, graph_output.size());
        ASSERT_EQ(1, poros_output.size());
        ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

/*
aten::instance_norm(Tensor input,
Tensor? weight,
Tensor? bias,
Tensor? running_mean,
Tensor? running_var,
bool use_input_stats,
float momentum,
float eps,
bool cudnn_enabled) -> Tensor
*/
TEST(Converters, ATenInstanceNormConvertsCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1: Tensor, %2: Tensor):
          %3 : NoneType = prim::Constant() 
          %4 : bool = prim::Constant[value=1]()
          %5 : float = prim::Constant[value=0.10000000000000001]()
          %6 : float = prim::Constant[value=1.0000000000000001e-05]()
          %7 : Tensor = aten::instance_norm(%0, %1, %2, %3, %3, %4, %5, %6, %4)
          return (%7))IR";

      auto input_tensor = at::randn({2, 10, 5, 5}, {at::kCUDA});
      auto weight = at::randn({10}, {at::kCUDA});
      auto bias = at::randn({10}, {at::kCUDA});

      baidu::mirana::poros::PorosOptions poros_option; // default device GPU
      baidu::mirana::poros::InstanceNormConverter instancenormconverter;

      // 运行原图与engine获取结果
      std::vector<at::Tensor> graph_output;
      std::vector<at::Tensor> poros_output;
      ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &instancenormconverter, 
                  {input_tensor, weight, bias}, graph_output, poros_output));

      ASSERT_EQ(1, graph_output.size());
      ASSERT_EQ(1, poros_output.size());
      ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));   
}

TEST(Converters, ATenInstanceNormConvertsNoWeightCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %3 : NoneType = prim::Constant() 
          %4 : bool = prim::Constant[value=1]()
          %5 : float = prim::Constant[value=0.10000000000000001]()
          %6 : float = prim::Constant[value=1.0000000000000001e-05]()
          %7 : Tensor = aten::instance_norm(%0, %3, %3, %3, %3, %4, %5, %6, %4)
          return (%7))IR";

      auto input_tensor = at::randn({2, 20, 45, 3}, {at::kCUDA});

      baidu::mirana::poros::PorosOptions poros_option; // default device GPU
      baidu::mirana::poros::InstanceNormConverter instancenormconverter;

      // 运行原图与engine获取结果
      std::vector<at::Tensor> graph_output;
      std::vector<at::Tensor> poros_output;
      ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &instancenormconverter, 
                  {input_tensor}, graph_output, poros_output));

      ASSERT_EQ(1, graph_output.size());
      ASSERT_EQ(1, poros_output.size());
      ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));   
}

TEST(Converters, ATenInstanceNormConverts3DCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %3 : NoneType = prim::Constant() 
          %4 : bool = prim::Constant[value=1]()
          %5 : float = prim::Constant[value=0.10000000000000001]()
          %6 : float = prim::Constant[value=1.0000000000000001e-05]()
          %7 : Tensor = aten::instance_norm(%0, %3, %3, %3, %3, %4, %5, %6, %4)
          return (%7))IR";

      auto input_tensor = at::randn({2, 20, 45}, {at::kCUDA});

      baidu::mirana::poros::PorosOptions poros_option; // default device GPU
      baidu::mirana::poros::InstanceNormConverter instancenormconverter;

      // 运行原图与engine获取结果
      std::vector<at::Tensor> graph_output;
      std::vector<at::Tensor> poros_output;
      ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &instancenormconverter, 
                  {input_tensor}, graph_output, poros_output));

      ASSERT_EQ(1, graph_output.size());
      ASSERT_EQ(1, poros_output.size());
      ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));   
}