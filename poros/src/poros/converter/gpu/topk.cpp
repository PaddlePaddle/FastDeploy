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
* @file topk.cpp
* @author tianjinjin@baidu.com
* @date Tue Sep  7 14:29:20 CST 2021
* @brief 
**/

#include "poros/converter/gpu/topk.h"
#include "poros/converter/gpu/weight.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/*
"aten::topk(Tensor self, 
int k, 
int dim=-1, 
bool largest=True, 
bool sorted=True) -> (Tensor values, Tensor indices)",
*/
bool TopkConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for TopkConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for TopkConverter is not Tensor as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto self_dim = nvdim_to_sizes(self->getDimensions());

    //extract k & dim & largest
    auto k = (engine->context().get_constant(inputs[1])).toInt();
    auto dim = (engine->context().get_constant(inputs[2])).toInt();
    auto largest = (engine->context().get_constant(inputs[3])).toBool();
    
    if (dim < 0) {
        dim = self_dim.size() + dim;
    }
    uint32_t shift_dim = 1 << dim;
    auto topk_type = largest ? (nvinfer1::TopKOperation::kMAX) : (nvinfer1::TopKOperation::kMIN);
    auto new_layer = engine->network()->addTopK(*self, topk_type, k, shift_dim);

    POROS_CHECK(new_layer, "Unable to create topk layer from node: " << *node);
    new_layer->setName((layer_info(node) + "_ITopKLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    engine->context().set_tensor(node->outputs()[1], new_layer->getOutput(1));
    LOG(INFO) << "Output tensor(0) shape: " << new_layer->getOutput(0)->getDimensions();
    LOG(INFO) << "Output tensor(1) shape: " << new_layer->getOutput(1)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, TopkConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
