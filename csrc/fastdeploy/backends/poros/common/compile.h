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
#include <algorithm>
#include <unordered_map>
#include <set>
#include <torch/script.h>

#include "iengine.h"
#include "poros_module.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief  compile graph
 *
 * @param [in] module : 原始module
 * @param [in] input_ivalues : 预热数据
 * @param [in] options : 参数
 * @return porosmodule
 * @retval !nullptr => succeed  nullptr => failed
 **/
std::unique_ptr<PorosModule> Compile(const torch::jit::Module& module, 
        const std::vector<std::vector<c10::IValue> >& prewarm_datas, 
        const PorosOptions& options);

class Compiler {
public:
    typedef std::unordered_map<const torch::jit::Node*, IEngine*> engine_map_t;
    typedef std::vector<std::vector<c10::IValue> > ivalue_vec_t;

    Compiler() : _origin_module(NULL) {}
    ~Compiler();

    /**
     * @brief initial Compiler
     *
     * @param [in] options : poros options
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    int init(const PorosOptions& options);

    /**
     * @brief compile whole graph
     *
     * @param [in] origin_module 
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [out] optimized_module : optimized graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    int compile(const torch::jit::Module& origin_module, 
                const ivalue_vec_t& prewarm_datas,
                torch::jit::Module* optimized_module);
    
private:

    /**
     * @brief preprocess this calculation graph
     *
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [out] graph : preprcessed graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    int preprocess_graph(const ivalue_vec_t& prewarm_datas, std::shared_ptr<torch::jit::Graph>& graph);

    /**
     * @brief segement this calculation graph
     *
     * @param [in/out] graph 
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    int segment_graph(std::shared_ptr<torch::jit::Graph>& graph);

    //分割子图（block)
    //分割后的子图，作为subgraph, 关联到block下。
    int segment_block(torch::jit::Block& block, IEngine* engine, int current_depth);
    
    //子图优化
    /**
     * @brief 子图优化
     *
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [in] opt_graph : ivalue_vec_t, vector of IValue
     * @param [out] optimized_module : optimized graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    int optimize_subgraph(const ivalue_vec_t& prewarm_datas, 
            const std::shared_ptr<torch::jit::Graph>& opt_graph,
            torch::jit::Module* optimized_module);

    //子图优化(block)
    int optimize_subblock(torch::jit::Block* block, 
            torch::jit::Module* optimized_module);

    /**
     * @brief 将子图基于engine编译成新图
     *
     * @param [in] engine : 子图用到的engine
     * @param [in] subgraph_node : 子图结点
     * @return [out] module : 转化后的模型
     * @retval 0 => succeed  <0 => failed
    **/
    int transform(IEngine* engine, torch::jit::Node& subgraph_node, 
            torch::jit::Module& module);

    /**
     * @brief 根据子图和options选择engine
     *
     * @param [in] node : 子图代表结点
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
    IEngine* select_engine(const torch::jit::Node* n);
    
    /**
     * @brief destory
     *
     * @return  void
    **/
    void close();

private:
    int _max_segment_depth{5};                    //最大子图分割深度
    ivalue_vec_t _prewarm_datas;                    //预热用的输入数据
    PorosOptions _options;
    engine_map_t _engine_map;                       //记录子图用的engine
    const torch::jit::Module* _origin_module;       //原始模型
    std::atomic<int> _engine_index = {0};            //记录engine的index
};

/**
 * @brief  compile graph, 内部使用
 *
 * @param [in] module : 原始module
 * @param [in] input_ivalues : 预热数据
 * @param [in] options : 参数
 * @return optimized_module
 * @retval !nullptr => succeed  nullptr => failed
 **/
std::unique_ptr<torch::jit::Module> CompileGraph(const torch::jit::Module& module, 
                                const std::vector<std::vector<c10::IValue> >& prewarm_datas, 
                                const PorosOptions& options);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu