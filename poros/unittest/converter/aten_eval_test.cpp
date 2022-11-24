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
* @file aten_eval_test.cpp
* @author wangrui39@baidu.com
* @date Mon December 13 11:36:11 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/aten_eval.h"
#include "poros/util/test_util.h"

static void aten_eval_test_helper(const std::string& graph_IR,
                            const std::vector<at::Tensor>& input_data,
                            baidu::mirana::poros::IConverter* converter) {
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


/*TEST(Converters, AppendConverterCorrectly) {
    //"aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))"

    const auto graph = R"IR(
        graph(%a.1 : Tensor,
          %b.1 : Tensor,
          %c.1 : Tensor):
          %12 : int = prim::Constant[value=0]()
          %x.1 : Tensor[] = prim::ListConstruct(%a.1)
          %7 : Tensor[] = aten::append(%x.1, %b.1) # test.py:26:4
          %10 : Tensor[] = aten::append(%x.1, %c.1) # test.py:27:4
          %13 : Tensor = aten::cat(%x.1, %12) # test.py:28:11
          return (%13))IR";

    std::vector<at::Tensor> input_data;
    auto input1 = at::randn({3, 4}, {at::kCUDA});
    auto input2 = at::randn({3, 4}, {at::kCUDA});
    auto input3 = at::randn({3, 4}, {at::kCUDA});
    
    input_data.push_back(input1);
    input_data.push_back(input2);
    input_data.push_back(input3);

    baidu::mirana::poros::AppendConverter appendConverter;
    aten_eval_test_helper(graph, input_data, &appendConverter);
}*/

TEST(Converters, GetitemConverterCorrectly) {
    // "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))"*/
    
    const auto graph = R"IR(
        graph(%a.1 : Tensor,
          %b.1 : Tensor):
          %12 : int = prim::Constant[value=0]() 
          %16 : int = prim::Constant[value=1]()
          %x.1 : Tensor[] = prim::ListConstruct(%a.1)
          %7 : Tensor[] = aten::append(%x.1, %b.1) 
          %ret.1 : Tensor = aten::__getitem__(%x.1, %12) 
          %17 : Tensor = aten::__getitem__(%x.1, %16) 
          %19 : Tensor = aten::add(%ret.1, %17, %16)
          return (%19))IR";

    std::vector<at::Tensor> input_data;
    auto input1 = at::randn({3, 4}, {at::kCUDA});
    auto input2 = at::randn({3, 4}, {at::kCUDA});
    
    input_data.push_back(input1);
    input_data.push_back(input2);

    baidu::mirana::poros::GetitemConverter getitemconverter;
    aten_eval_test_helper(graph, input_data, &getitemconverter);
}

TEST(Converters, SetitemConverterCorrectly) {
    // aten::_set_item.t(t[](a!) l, int idx, t(b -> *) el) -> (t[](a!))
    const auto graph = R"IR(
        graph(%x.1 : Tensor,
          %y.1 : Tensor):
          %6 : int = prim::Constant[value=1]() # test.py:28:15
          %10 : int = prim::Constant[value=0]() # test.py:28:6
          %a.1 : Tensor[] = prim::ListConstruct(%x.1, %y.1)
          %8 : Tensor = aten::add(%x.1, %6, %6) # test.py:28:11
          %11 : Tensor[] = aten::_set_item(%a.1, %10, %8) # test.py:28:4
          %ret.1 : Tensor = aten::cat(%a.1, %10) # test.py:29:10
          return (%ret.1))IR";

    std::vector<at::Tensor> input_data;
    auto input1 = at::randn({3, 4}, {at::kCUDA});
    auto input2 = at::randn({3, 4}, {at::kCUDA});
    
    input_data.push_back(input1);
    input_data.push_back(input2);

    baidu::mirana::poros::SetitemConverter setitemconverter;
    aten_eval_test_helper(graph, input_data, &setitemconverter);
}

static void eval_dynamic_test_helper(const std::string& graph_IR, 
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

TEST(Converters, ATenGetitemdynamicConvertsCorrectly) {
    // "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))"*/
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : int = prim::Constant[value=1]()
          %3 : int = aten::__getitem__(%1, %2) 
          %4 : Tensor = aten::add(%0, %3, %2)
          return (%4))IR";
    baidu::mirana::poros::GetitemConverter getitemconverter;
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({4, 5}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[1].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));
    prewarm_data[2].push_back(at::zeros({2, 3}, {at::kCUDA}).to(at::ScalarType::Int));

    eval_dynamic_test_helper(graph_IR, &getitemconverter, input_data, true, &prewarm_data);
}
