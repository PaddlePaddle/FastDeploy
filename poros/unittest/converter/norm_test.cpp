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
* @file norm_test.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-23 20:38:15
* @brief
**/

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/norm.h"
#include "poros/util/test_util.h"

static void norm_test_helper(const std::string &graph_IR,
                             baidu::mirana::poros::IConverter *converter,
                             std::vector<int64_t> shape1 = {5}) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape1, {at::kCUDA}));

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

static std::string gen_norm_tensor_graph(const std::string &p, const std::string &dims, const std::string &keepdim) {
    return R"IR(
graph(%1 : Tensor):
  %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
  %3 : int = prim::Constant[value=)IR" + p + R"IR(]()
  %4 : int[] = prim::Constant[value=)IR" + dims + R"IR(]()
  %5 : Tensor = aten::norm(%1, %3, %4, %2)
  return (%5)
)IR";
}

static std::string gen_norm_empty_dims_graph(const std::string &p, const std::string &dims, const std::string &keepdim) {
    return R"IR(
graph(%1 : Tensor):
  %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
  %3 : int = prim::Constant[value=)IR" + p + R"IR(]()
  %4 : int[] = prim::ListConstruct()
  %5 : Tensor = aten::norm(%1, %3, %4, %2)
  return (%5)
)IR";
}



TEST(Converters, ATenNormConvertsCorrectlyWith) {
    std::vector<std::string> graphIRs;
    graphIRs.push_back(gen_norm_tensor_graph("2", "[0]","0"));
    graphIRs.push_back(gen_norm_tensor_graph("2", "[1]","0"));
    graphIRs.push_back(gen_norm_empty_dims_graph("2", "","0"));
    graphIRs.push_back(gen_norm_tensor_graph("2", "[1,2]","0"));
    graphIRs.push_back(gen_norm_tensor_graph("2", "[-2,2]","0"));
    graphIRs.push_back(gen_norm_tensor_graph("2", "[1,2]","1"));
    graphIRs.push_back(gen_norm_tensor_graph("1.5", "[1,2]","0"));
    graphIRs.push_back(gen_norm_tensor_graph("0.2", "[-1,-2,-3,-4]","1"));

    baidu::mirana::poros::NormConverter converter;

    for(auto ir:graphIRs){
        norm_test_helper(ir, &converter, {3,4,5,6,7});
    }
}

static std::string gen_frobenius_norm_tensor_graph(const std::string &dims, const std::string &keepdim) {
    return R"IR(
graph(%1 : Tensor):
  %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
  %4 : int[] = prim::Constant[value=)IR" + dims + R"IR(]()
  %5 : Tensor = aten::frobenius_norm(%1, %4, %2)
  return (%5)
)IR";
}

static std::string gen_frobenius_norm_empty_dims_graph(const std::string &dims, const std::string &keepdim) {
    return R"IR(
graph(%1 : Tensor):
  %2 : bool = prim::Constant[value=)IR" + keepdim + R"IR(]()
  %4 : int[] = prim::ListConstruct()
  %5 : Tensor = aten::frobenius_norm(%1, %4, %2)
  return (%5)
)IR";
}

TEST(Converters, ATenFrobeniusNormConvertsCorrectlyWith) {
    std::vector<std::string> graphIRs;
    graphIRs.push_back(gen_frobenius_norm_tensor_graph( "[0]","0"));
    graphIRs.push_back(gen_frobenius_norm_tensor_graph("[1]","0"));
    graphIRs.push_back(gen_frobenius_norm_empty_dims_graph("","0"));
    graphIRs.push_back(gen_frobenius_norm_tensor_graph("[1,2]","0"));
    graphIRs.push_back(gen_frobenius_norm_tensor_graph("[-2,2]","0"));
    graphIRs.push_back(gen_frobenius_norm_tensor_graph("[1,2]","1"));
    graphIRs.push_back(gen_frobenius_norm_tensor_graph( "[1,2]","0"));

    baidu::mirana::poros::FrobeniusNormConverter converter;

    for(auto ir:graphIRs){
        norm_test_helper(ir, &converter, {3,4,5,6,7});
    }
}