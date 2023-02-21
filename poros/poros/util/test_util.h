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
#include "poros/converter/iconverter.h"

namespace baidu {
namespace mirana {
namespace poros {
namespace testutil {
/**
 * @brief run graph and poros, then compare their outputs
 *
 * @param [in] graph_IR : string of IR
 * @param [in] poros_option : default device is GPU
 * @param [in] converter : converter tested
 * @param [in] input_data : vector of at::Tensor, once input data of graph
 * @param [in] log_path : record the running time of the graph and engine. default is none and don't record.
 * @param [in] prewarm_data : preheating data, default is null and input_data is used for preheating
 * @param [in] const_input_indices : the data index in input_data, which will trans to constant-tensor before graph run. 
 *                          (ie. constant weight parameter), this can change the graph and real input datas implicitly.
 * @param [out] graph_output : vector of at::Tensor, graph outputs
 * @param [out] poros_output : vector of at::Tensor, poros outputs
 * @return  bool
 * @retval true => succeed  false => failed
**/
bool run_graph_and_poros(const std::string &graph_IR,
                         const baidu::mirana::poros::PorosOptions &poros_option,
                         baidu::mirana::poros::IConverter *converter,
                         const std::vector<at::Tensor> &input_data,
                         std::vector<at::Tensor> &graph_output,
                         std::vector<at::Tensor> &poros_output,
                         const std::vector<std::vector<at::Tensor>> *prewarm_data = nullptr,
                         std::string log_path = "",
                         const std::vector<size_t> const_input_indices = {});

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

}// namespace testutil
}// namespace poros
}// namespace mirana
}// namespace baidu
