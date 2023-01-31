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
* @file gpu_converter.h
* @author tianjinjin@baidu.com
* @author huangben@baidu.com
* @date Tue Jul 27 11:24:21 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include "torch/script.h"

#include "poros/converter/iconverter.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/log/poros_logging.h"

namespace baidu {
namespace mirana {
namespace poros {

class GpuConverter : public IConverter {
public:
    virtual ~GpuConverter() {}
    virtual bool converter(TensorrtEngine* engine, const torch::jit::Node *node) = 0;
    virtual bool converter(IEngine* engine, const torch::jit::Node *node) {
        return converter(static_cast<TensorrtEngine*>(engine), node);
    }
    virtual const std::vector<std::string> schema_string() = 0;
    virtual const std::vector<torch::jit::NodeKind> node_kind() = 0;
    
protected:
    /**
     * @brief Check whether the scalar inputs of the node are nvinfer1::ITensor (not come from prim::Constant). 
     *        If yes, convert other scalar inputs to nvinfer1::ITensor and save them in _tensor_scalar_map. 
     *        The type of nvinfer1::ITensor is consistent with the original scalar.
     *
     * @param [in] engine : TensorrtEngine
     * @param [in] node : node in torch::jit::Graph
     * @return bool
     * @retval true => yes  false => no
    **/
    bool check_inputs_tensor_scalar(TensorrtEngine* engine, const torch::jit::Node *node);
    /**
     * @brief get nvinfer1::ITensor* type scalar from _tensor_scalar_map.
     *
     * @param [in] value : the input value of the node.
     * @return nvinfer1::ITensor*
    **/
    nvinfer1::ITensor* get_tensor_scalar(const torch::jit::Value* value);

private:
    std::unordered_map<const torch::jit::Value*, nvinfer1::ITensor*> _tensor_scalar_map;
};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
