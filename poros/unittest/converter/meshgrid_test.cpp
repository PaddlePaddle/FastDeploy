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
* @file meshgrid_test.cpp
* @author wangrui39@baidu.com
* @date Monday November 27 11:36:11 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/meshgrid.h"
#include "poros/util/test_util.h"

static void add_test_helper(const std::string& graph_IR, 
                            baidu::mirana::poros::IConverter* converter,
                            std::vector<float> value1 = {1.0, 2.0, 3.0},
                            std::vector<float> value2 = {4.0, 5.0}){
    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0);//.dtype(torch::kInt32);
    input_data.push_back(at::tensor(value1, options_pyt));
    input_data.push_back(at::tensor(value2, options_pyt));

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

static std::string gen_meshgrid_graph() {
    std::string graph = R"IR(
      graph(%x.1 : Tensor,
        %y.1 : Tensor):
        %10 : int = prim::Constant[value=1]()
        %4 : Tensor[] = prim::ListConstruct(%x.1, %y.1)
        %5 : Tensor[] = aten::meshgrid(%4)
        %grid_x.1 : Tensor, %grid_y.1 : Tensor = prim::ListUnpack(%5)
        %11 : Tensor = aten::add(%grid_x.1, %grid_y.1, %10)
        return (%11))IR"; 

    return graph;
}

TEST(Converters, ATenMeshgridConvertsCorrectly) {
    const auto graph_IR = gen_meshgrid_graph();
    baidu::mirana::poros::MeshgridConverter meshgridconverter;
    add_test_helper(graph_IR, &meshgridconverter);
}