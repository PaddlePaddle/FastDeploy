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
* @file activation_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/activation.h"
#include "poros/util/test_util.h"

static void activation_test_helper(const std::string& graph_IR, 
                                  baidu::mirana::poros::IConverter* converter,
                                  std::vector<int64_t> shape1 = {5},
                                  bool single_input = true,
                                  std::vector<int64_t> shape2 = {5}){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));
    if (!single_input){
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
    std::string gelu_node_str("aten::gelu");
    if (converter->node_kind()[0].toQualString() == gelu_node_str){
        // NOTE: The official tensorrt plugin applies the Gelu activation x * Phi(x), where Phi is the Gaussian cdf,
        // approximated by: 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x^3))) and the pytorch uses
        // c10::cuda::compat::normcdf to compute Phi(x). So there's a difference here and therefore the threshold is slightly
        // higher than other ops. One in ten runs will give you an out of normal threshold result.
        ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 5e-2));
    }else{
        ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    }
}

static std::string gen_single_node_graph(const std::string& op) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : Tensor = aten::)IR" + op + R"IR((%0)
          return (%1))IR";
}

static std::string gen_hardtanh_graph(const std::string& op, 
                                      const std::string& min_val, 
                                      const std::string& max_val) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : float = prim::Constant[value=)IR" + min_val + R"IR(]()
          %2 : float = prim::Constant[value=)IR" + max_val + R"IR(]()
          %3 : Tensor = aten::)IR" + op + R"IR((%0, %1, %2)
          return (%3))IR";
}
static std::string gen_leakyrelu_graph(const std::string& op, const std::string& negative_slope) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : float = prim::Constant[value=)IR" + negative_slope + R"IR(]()
          %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
          return (%2))IR";
}

static std::string gen_elu_graph(const std::string& alpha) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : float = prim::Constant[value=)IR" + alpha + R"IR(]()
          %2 : int = prim::Constant[value=1]()
          %3 : Tensor = aten::elu(%0, %1, %2, %2)
          return (%3))IR";
}

TEST(Converters, ATenReluConvertsCorrectly) {
    // aten::relu(Tensor self) -> Tensor
    const auto graph_IR = gen_single_node_graph("relu");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenRelu_ConvertsCorrectly) {
    // aten::relu_(Tensor(a!) self) -> Tensor(a!)
   const auto graph_IR = gen_single_node_graph("relu_");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenRelu6ConvertsCorrectly) {
    // aten::relu6(Tensor self) -> Tensor
    const auto graph_IR = gen_single_node_graph("relu6");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenRelu6_ConvertsCorrectly) {
    // aten::relu6_(Tensor(a!) self) -> Tensor(a!)
    const auto graph_IR = gen_single_node_graph("relu6_");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenSigmoidConvertsCorrectly) {
    // aten::sigmoid(Tensor self) -> Tensor
    const auto graph_IR = gen_single_node_graph("sigmoid");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenSigmoid_ConvertsCorrectly) {
    // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
    const auto graph_IR = gen_single_node_graph("sigmoid_");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenTanhConvertsCorrectly) {
    // aten::tanh(Tensor self) -> Tensor"
    const auto graph_IR = gen_single_node_graph("tanh");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenTanh_ConvertsCorrectly) {
    // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
    const auto graph_IR = gen_single_node_graph("tanh_");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenGeluConvertsCorrectly) {
    std::string graph_IR_str;
    // aten::gelu schema changed in torch-1.12
    if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 12) {
        // aten::gelu(Tensor self) -> Tensor
        graph_IR_str = gen_single_node_graph("gelu");
    } else {
        // aten::gelu(Tensor self, *, str approximate='none') -> Tensor
        graph_IR_str = R"IR(
            graph(%0 : Tensor):
                %approximate : str = prim::Constant[value="tanh"]()
                %1 : Tensor = aten::gelu(%0, %approximate)
                return (%1))IR";
    }
    const auto graph_IR = graph_IR_str;
    baidu::mirana::poros::GeluActivationConverter geluactivationconverter;
    activation_test_helper(graph_IR, &geluactivationconverter, {10});
}

TEST(Converters, ATenLeakyreluConvertsCorrectly) {
    // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    const auto graph_IR = gen_leakyrelu_graph("leaky_relu", "0.01");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenLeakyreluNegSlopeConvertsCorrectly) {
    // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    const auto graph_IR = gen_leakyrelu_graph("leaky_relu", "0.05");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenHardtanhConvertsCorrectly) {
    // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
    const auto graph_IR = gen_hardtanh_graph("hardtanh", "-1.0", "1.0");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenHardtanhMinvalMaxvalConvertsCorrectly) {
    // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
    const auto graph_IR = gen_hardtanh_graph("hardtanh", "-3.5", "2.5");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenHardtanh_ConvertsCorrectly) {
    // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
    const auto graph_IR = gen_hardtanh_graph("hardtanh_", "-1.0", "1.0");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenHardtanh_MinvalMaxvalConvertsCorrectly) {
    // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
    const auto graph_IR = gen_hardtanh_graph("hardtanh_", "-2.1", "3.8");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenEluConvertsCorrectly) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    const auto graph_IR = gen_elu_graph("1.0");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenEluAlphaConvertsCorrectly) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    const auto graph_IR = gen_elu_graph("3.4");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}

TEST(Converters, ATenEluNegAlphaConvertsCorrectly) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    const auto graph_IR = gen_elu_graph("-2.1");
    baidu::mirana::poros::ActivationConverter activationconverter;
    activation_test_helper(graph_IR, &activationconverter);
}