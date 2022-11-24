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
* @file tensorrt_engine.h
* @author huangben@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#pragma once

//from cuda
#include <cuda_runtime.h>

//from pytorch
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

//from tensorrt
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "poros/compile/poros_module.h"
#include "poros/engine/engine_context.h"
#include "poros/engine/iengine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/log/tensorrt_logging.h"

namespace baidu {
namespace mirana {
namespace poros {

/**
 * the implement of tensorRT engine
 **/

class TensorrtEngine : public IEngine {
public:
    TensorrtEngine();
    TensorrtEngine(std::string engine_str);
    //virtual ~TensorrtEngine();

    /**
     * @brief init
     * @return int
     * @retval 0 => success, <0 => fail
     **/
    virtual int init() override;

    /**
     * @brief 核心实现
     *        编译期将subgraph转化成对应engine的图结构保存在engine内部，以使得运行期的excute_engine能调用, 此处保证所有的op都被支持，
     * @      注意要注册输入输出tensor到engine_context中
     * @param [in] sub_graph  : 子图
     * @return [res]int
     * @retval 0 => success, <0 => fail
     **/
    virtual int transform(const PorosGraph& sub_graph) override;

    /**
     * @brief 子图执行期逻辑
     * @param [in] inputs  : 输入tensor
     * @return [res] 输出tensor
     **/
    virtual std::vector<at::Tensor> excute_engine(const std::vector<at::Tensor>& inputs) override;

    /**
     * @brief 在jit 模块中标记engine
     * @param [in] name : engine sign
     * @param [in] module  : jit modeule
     * @param [out] module  : 添加了engine sign之后的module
     **/
    virtual void register_module_attribute(const std::string& name, torch::jit::Module& module) override;

    /**
     * @brief get engine mark
     * @retval engine name
    **/
    virtual const std::string who_am_i() override {
        return "TensorrtEngine";
    }

    /**
     * @brief get context
     * @retval context
    **/
    EngineContext<nvinfer1::ITensor>& context() {
        return _context;
    }

    /**
     * @brief get network
     * @retval network
    **/
    nvinfer1::INetworkDefinition* network() {
        return _network.get();
    }

    /**
     * @brief get cuda engine
     * @retval engine
    **/
    nvinfer1::ICudaEngine* cuda_engine() {
        return _cuda_engine.get();
    }  

private:

    /**
     * @brief convert input type from torch to tensorrt
     * @param [in] input : input value
     * @param [in] input_type : input valur type
    **/
    //DEPRECATED
    void gen_tensorrt_input_type(const torch::jit::Value* input,
                                nvinfer1::DataType& input_type);

    /**
     * @brief extract input value from subgraph_node
     * @param [in] sub_graph : poros graph
     * @retval 0 => success, <0 => fail
    **/
    int init_engine_inputs(const PorosGraph& sub_graph);

    /**
     * @brief mark a tensor as a network output.
     * @param [in] outputs : outputs value list
     * @retval 0 => success, <0 => fail
    **/
    int mark_graph_outputs(at::ArrayRef<const torch::jit::Value*> outputs);
    
    /**
     * @brief binding input and output for engine
    **/ 
    void binding_io();

    /**
     * @brief convert jit graph to engine
     * @param [in] graph : jit grpah
     * @retval tetengin serialize data
    **/
   //DEPRECATED
    std::string convert_graph_to_engine(std::shared_ptr<torch::jit::Graph>& graph);

    /**
     * @brief gen dynamic dims for given value
     * @param [in] value : jit value
     * @retval value dims
    **/
    nvinfer1::Dims gen_dynamic_dims(torch::jit::Value* value);
    
private:
    //for tensortrt networkbuilding
    baidu::mirana::poros::TensorrtLogger _logger;
    std::shared_ptr<nvinfer1::IBuilder> _builder;
    std::shared_ptr<nvinfer1::INetworkDefinition> _network;
    std::shared_ptr<nvinfer1::IBuilderConfig> _cfg;

    //engine conrtext. to store the relationship of value-itensor
    baidu::mirana::poros::EngineContext<nvinfer1::ITensor> _context;

    //for runtime
    std::shared_ptr<nvinfer1::IRuntime> _runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> _cuda_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> _exec_ctx;

    std::unordered_map<uint64_t, uint64_t> _in_binding_map;
    std::unordered_map<uint64_t, uint64_t> _out_binding_map;

    std::unordered_set<uint64_t> _in_shape_tensor_index;

    PorosOptions _poros_options;
    std::shared_ptr<std::mutex> _mutex; //for enqueue
};

//std::vector<at::Tensor> execute_engine(const std::vector<at::Tensor> inputs,
//                            c10::intrusive_ptr<TensorrtEngine> compiled_engine);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
