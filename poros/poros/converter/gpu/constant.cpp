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
* @file constant.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/
#include "torch/script.h"

#include "poros/converter/gpu/constant.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/context/poros_global.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

bool ConstantConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    c10::optional<torch::jit::IValue> ivalue = toIValue(node->output());
    POROS_CHECK_TRUE(ivalue.has_value(), "invaid data for ConstantConverter");
    engine->context().set_constant(node->outputs()[0], ivalue.value());

    //situation1: Tensor
    if (ivalue.value().isTensor()) {
        auto tensor = ivalue.value().toTensor();
        auto t_weights = Weights(tensor);
        auto const_layer = engine->network()->addConstant(t_weights.shape, t_weights.data);
        const_layer->setName(layer_info(node).c_str());
        engine->context().set_tensor(node->outputs()[0], const_layer->getOutput(0));
    }
    //situation2: Tensor[]
    else if(ivalue.value().isTensorList()) {
        auto c10_tensorlist = ivalue.value().toTensorList();
        std::vector<nvinfer1::ITensor*> tensorlist;
        tensorlist.reserve(c10_tensorlist.size());
        for (size_t i = 0; i < c10_tensorlist.size(); i++){
            nvinfer1::ITensor* nv_tensor = tensor_to_const(engine, c10_tensorlist[i]);
            tensorlist.emplace_back(nv_tensor);
        }
        engine->context().set_tensorlist(node->outputs()[0], tensorlist);
    }
    //situation3: Tensor?[]
    else if (ivalue.value().type()->str().find("Tensor?[]") != std::string::npos) {
        c10::List<c10::IValue> c10_tensorlist = ivalue.value().to<c10::List<c10::IValue>>();
        std::vector<nvinfer1::ITensor*> tensorlist;
        tensorlist.reserve(c10_tensorlist.size());
        for (size_t i = 0; i < c10_tensorlist.size(); i++){
            auto tensor = c10_tensorlist.get(i).toTensor();
            nvinfer1::ITensor* nv_tensor = tensor_to_const(engine, tensor);
            tensorlist.emplace_back(nv_tensor);
        }
        engine->context().set_tensorlist(node->outputs()[0], tensorlist);
    }

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ConstantConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
