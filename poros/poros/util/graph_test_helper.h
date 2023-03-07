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
* @file test_util.h
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#pragma once

#include "poros/compile/poros_module.h"
#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {
namespace graphtester {


enum InputTypeEnum {
    InputTensor = 0, // op 的输入
    ConstantTensor, // op的权重和偏置
    ConstantIntVector, // 如conv2d的stride等要求int[]的参数
};

/**
 *
 * @param graph_IR
 * @param poros_option : default device is GPU
 * @param fuser : op fuser to test
 * @param input_data : vector of at::IValue, which Compatible with Tensor, vector<int64_t/double_t ...> , scalar , etc.
 * @param input_data_type_mask : tell the func how to deal with the input data, used to trans Tensor, vector or []int to prim::Constant
 * @param original_graph_output : vector of at::Tensor, graph outputs
 * @param fused_graph_output : vector of at::Tensor, poros outputs
 * @param log_path
 * @return bool
 */
bool run_graph_and_fused_graph(const std::string &graph_IR,
                               const baidu::mirana::poros::PorosOptions &poros_option,
                               std::shared_ptr<baidu::mirana::poros::IFuser> fuser,
                               const std::vector<at::IValue> &input_data,
                               const std::vector<InputTypeEnum> &input_data_type_mask,
                               std::vector<at::Tensor> &original_graph_output,
                               std::vector<at::Tensor> &fused_graph_output,
                               std::string log_path = "");

/**
 * @brief compare the similarity of two Tensors containing Float
 *
 * @param [in] a : first Tensor
 * @param [in] b : second Tensor
 * @param [in] threshold : acceptable relative threshold
 * @return  bool
 * @retval true => succeed  false => failed
**/
bool almost_equal(const at::Tensor &a, const at::Tensor &b, const float &threshold);

}// namespace graphtester
}// namespace poros
}// namespace mirana
}// namespace baidu
