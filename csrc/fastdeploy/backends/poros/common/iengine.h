// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>

//from pytorch
#include <torch/script.h>
#include <torch/csrc/jit/ir/ir.h>
#include <ATen/core/interned_strings.h>

#include "plugin_create.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * the base engine class 
 * every registered engine should inherit from this IEngine
 **/

struct PorosGraph {
    torch::jit::Graph* graph = NULL;
    torch::jit::Node* node = NULL;
};

typedef uint64_t EngineID;

class IEngine : public IPlugin, public torch::CustomClassHolder{
public:
    virtual ~IEngine() {}

    /**
     * @brief init, 必须init成功才算初始化成功
     * @return int
     * @retval 0 => success, <0 => fail
     **/
    virtual int init() = 0;
    
    /**
     * @brief 编译期将subgraph转化成对应engine的图结构保存在engine内部，以使得运行期的excute_engine能调用, 此处保证所有的op都被支持，核心实现
     * @param [in] sub_graph  : 子图
     * @return [res]int
     * @retval 0 => success, <0 => fail
     **/
    virtual int transform(const PorosGraph& sub_graph) = 0;

    /**
     * @brief 子图执行期逻辑
     * @param [in] inputs  : 输入tensor
     * @return [res] 输出tensor
     **/
    virtual std::vector<at::Tensor> excute_engine(const std::vector<at::Tensor>& inputs) = 0;

    virtual void register_module_attribute(const std::string& name, torch::jit::Module& module) = 0;

    //标识
    virtual const std::string who_am_i() = 0;

    //node是否被当前engine支持
    bool is_node_supported(const torch::jit::Node* node);

public:
    std::pair<uint64_t, uint64_t> _num_io; //输入/输出参数个数
    EngineID _id;

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
