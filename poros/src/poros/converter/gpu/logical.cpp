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
* @file logical.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-17 18:32:04
* @brief
**/

#include "poros/converter/gpu/logical.h"
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

bool AndConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for AndConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for AndConverter is not Tensor as expected");
    POROS_CHECK_TRUE(((inputs[0]->node()->kind() != torch::jit::prim::Constant) &&
        (inputs[1]->node()->kind() != torch::jit::prim::Constant)),
                     "constant input is not support for AndConverter");

    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    auto other = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE((other != nullptr), "Unable to init input tensor for node: " << *node);

    POROS_CHECK_TRUE(((self->getType() == nvinfer1::DataType::kBOOL) && (other->getType() == nvinfer1::DataType::kBOOL)),
                     "Only Bool type supported for for node: " << *node);

    POROS_CHECK_TRUE(((self->getDimensions().nbDims > 0) && (other->getDimensions().nbDims > 0)),
                     "scalar input is not supported for node: " << *node);

    auto new_layer = add_elementwise(engine,
            nvinfer1::ElementWiseOperation::kAND,
            self,
            other,
            layer_info(node) + "_and");

    POROS_CHECK(new_layer, "Unable to create And layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}


bool OrConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for OrConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for OrConverter is not Tensor as expected");
    POROS_CHECK_TRUE(((inputs[0]->node()->kind() != torch::jit::prim::Constant) &&
                      (inputs[1]->node()->kind() != torch::jit::prim::Constant)),
                     "constant input is not support for OrConverter");

    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    auto other = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE((other != nullptr), "Unable to init input tensor for node: " << *node);

    POROS_CHECK_TRUE(((self->getType() == nvinfer1::DataType::kBOOL) && (other->getType() == nvinfer1::DataType::kBOOL)),
                     "Only Bool type supported for for node: " << *node);

    POROS_CHECK_TRUE(((self->getDimensions().nbDims > 0) && (other->getDimensions().nbDims > 0)),
                     "scalar input is not supported for node: " << *node);

    auto new_layer = add_elementwise(engine,
                                     nvinfer1::ElementWiseOperation::kOR,
                                     self,
                                     other,
                                     layer_info(node) + "_or");

    POROS_CHECK(new_layer, "Unable to create Or layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

bool XorConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for XorConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for XorConverter is not Tensor as expected");
    POROS_CHECK_TRUE(((inputs[0]->node()->kind() != torch::jit::prim::Constant) &&
                      (inputs[1]->node()->kind() != torch::jit::prim::Constant)),
                     "constant input is not support for XorConverter");

    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    auto other = engine->context().get_tensor(inputs[1]);
    POROS_CHECK_TRUE((other != nullptr), "Unable to init input tensor for node: " << *node);

    POROS_CHECK_TRUE(((self->getType() == nvinfer1::DataType::kBOOL) && (other->getType() == nvinfer1::DataType::kBOOL)),
                     "Only Bool type supported for for node: " << *node);

    POROS_CHECK_TRUE(((self->getDimensions().nbDims > 0) && (other->getDimensions().nbDims > 0)),
                     "scalar input is not supported for node: " << *node);

    auto new_layer = add_elementwise(engine,
                                     nvinfer1::ElementWiseOperation::kXOR,
                                     self,
                                     other,
                                     layer_info(node) + "_xor");

    POROS_CHECK(new_layer, "Unable to create Xor layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

bool NotConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for NotConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for NotConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[0]->node()->kind() != torch::jit::prim::Constant),
                     "constant input is not support for NotConverter");

    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto new_layer = engine->network()->addUnary(*self,nvinfer1::UnaryOperation::kNOT);

    POROS_CHECK(new_layer, "Unable to create And layer from node: " << *node);
    new_layer->setName((layer_info(node) + "_IUnaryLayer").c_str());

    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, AndConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, OrConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, XorConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, NotConverter);

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
