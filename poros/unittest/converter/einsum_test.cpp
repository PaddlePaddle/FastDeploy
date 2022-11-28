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
* @file einsum_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Jul 06 11:24:51 CST 2022
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/einsum.h"
#include "poros/util/test_util.h"

static void aten_einsum_test_helper(const std::string& equation, 
                                    at::Tensor input1, 
                                    at::Tensor input2 = at::Tensor()) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(input1);
    if (input2.defined()) {
        input_data.push_back(input2);
    }
    
    std::string graph_IR;
    if (input_data.size() == 2) {
        graph_IR = R"IR(
            graph(%0 : Tensor, %1 : Tensor):
                %eq : str = prim::Constant[value=")IR" + equation + R"IR("]()
                %2 : Tensor[] = prim::ListConstruct(%0, %1)
                %3 : Tensor = aten::einsum(%eq, %2)
                return (%3))IR";
    } else {
        graph_IR = R"IR(
            graph(%0 : Tensor):
                %eq : str = prim::Constant[value=")IR" + equation + R"IR("]()
                %2 : Tensor[] = prim::ListConstruct(%0)
                %3 : Tensor = aten::einsum(%eq, %2)
                return (%3))IR";
    }
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::EinsumConverter einsumconverter;

    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &einsumconverter, 
                input_data, graph_output, poros_output));
    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenEinsumConverterCorrectly) {
// aten::einsum(str equation, Tensor[] tensors) -> (Tensor)
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %eq : str = prim::Constant[value="bfnd,ndh->bfh"]()
        %2 : Tensor[] = prim::ListConstruct(%0, %1)
        %3 : Tensor = aten::einsum(%eq, %2)
        return (%3))IR";

    std::vector<at::Tensor> input_data;
    
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::randn({20, 30, 12, 26}, options_pyt_float));
    input_data.push_back(at::randn({12, 26, 312}, options_pyt_float));

    baidu::mirana::poros::EinsumConverter einsumconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &einsumconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

TEST(Converters, ATenEinsumTorchExamplesTestConverterCorrectly) {
    // Test cases from https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    at::Tensor x = at::randn({5}, options_pyt_float);
    at::Tensor y = at::randn({7}, options_pyt_float);
    at::Tensor A = at::randn({3, 5}, options_pyt_float);
    at::Tensor B = at::randn({2, 5}, options_pyt_float);
    at::Tensor C = at::randn({2, 3, 5}, options_pyt_float);
    at::Tensor D = at::randn({2, 5, 7}, options_pyt_float);
    at::Tensor E = at::randn({7, 9}, options_pyt_float);
    at::Tensor F = at::randn({2, 3, 3, 5}, options_pyt_float);
    at::Tensor G = at::randn({5, 4, 6}, options_pyt_float);
    at::Tensor H = at::randn({4, 4}, options_pyt_float);
    at::Tensor I = at::randn({2, 3, 2}, options_pyt_float);

    // vector operations
    aten_einsum_test_helper("i->", x);                        // sum
    aten_einsum_test_helper("i,i->", x, x);                   // dot
    aten_einsum_test_helper("i,i->i", x, x);                  // vector element-wisem mul
    aten_einsum_test_helper("i,j->j", x, y);                  // outer

    // Matrix operations
    aten_einsum_test_helper("ij->ji", A);                     // transpose
    aten_einsum_test_helper("ij->j", A);                      // row sum
    aten_einsum_test_helper("ij->i", A);                      // col sum
    aten_einsum_test_helper("ij,ij->ij", A, A);               // matrix element-wise mul
    aten_einsum_test_helper("ij,j->i", A, x);                 // matrix vector multiplication
    aten_einsum_test_helper("ij,kj->ik", A, B);               // matmul
    aten_einsum_test_helper("ij,ab->ijab", A, E);             // matrix outer product

    // Tensor operations
    aten_einsum_test_helper("Aij,Ajk->Aik", C, D);            // batch matmul
    aten_einsum_test_helper("ijk,jk->i", C, A);               // tensor matrix contraction
    aten_einsum_test_helper("aij,jk->aik", D, E);             // tensor matrix contraction
    aten_einsum_test_helper("abCd,dfg->abCfg", F, G);         // tensor tensor contraction
    aten_einsum_test_helper("ijk,jk->ik", C, A);              // tensor matrix contraction with double indices
    aten_einsum_test_helper("ijk,jk->ij", C, A);              // tensor matrix contraction with double indices
    aten_einsum_test_helper("ijk,ik->j", C, B);               // non contiguous
    aten_einsum_test_helper("ijk,ik->jk", C, B);              // non contiguous with double indices

    // Diagonal operations are not permitted in poros
    // aten_einsum_test_helper("ii", H);                          // trace
    // aten_einsum_test_helper("ii->i", H);                       // diagonal
    // aten_einsum_test_helper("iji->j", I);                      // non-contiguous trace
    // aten_einsum_test_helper("ngrg...->nrg...", at::randn({2, 1, 3, 1, 4}, options_pyt_float));

    // Ellipsis equations are not permitted in poros
    // aten_einsum_test_helper("i...->...", H);
    // aten_einsum_test_helper("ki,...k->i...", A.t(), B);
    // aten_einsum_test_helper("k...,jk->...", A.t(), B);
    // aten_einsum_test_helper('...ik, ...j -> ...ij', C, x);
    // aten_einsum_test_helper('Bik,k...j->i...j', C, at::randn({5, 3}, options_pyt_float));
    // aten_einsum_test_helper('i...j, ij... -> ...ij', C, at::randn({2, 5, 2, 3}, options_pyt_float));
}