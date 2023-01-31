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
* @file norm.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-02-23 20:33:41
* @brief
**/

#include <cmath>
#include "poros/converter/gpu/norm.h"
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

bool NormConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    // inputs.size() == 4
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for NormConverter")

    // inputs[0] => self
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for NormConverter is not Tensor as expected");
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node)
    auto self_dims = self->getDimensions();

    // inputs[1] => p
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
                     "Non-constant p is not support for NormConverter");
    auto p_const = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((p_const.isScalar()), "Non-scalar p is not support for NormConverter")
    nvinfer1::ITensor *p, *p_inverse;
    auto p_scalar = p_const.toScalar().to<float>();
    p = tensor_to_const(engine, torch::tensor(p_scalar));
    p_inverse = tensor_to_const(engine, torch::tensor(1.0 / p_scalar));

    // inputs[2] => dims
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
                     "Non-constant dims is not support for NormConverter");
    auto dims_const = engine->context().get_constant(inputs[2]);
    POROS_CHECK_TRUE((dims_const.isIntList()), " dims type must be int[] for NormConverter")
    auto dims_list = dims_const.toIntList().vec();
    uint32_t dim = 0;
    for (auto d: dims_list) {
        if (d < 0) {
            d = self_dims.nbDims + d;
        }
        dim |= 1 << d;
    }
    if (dim == 0) {
        dim = pow(2, self_dims.nbDims) - 1;
    }

    // input[3] => keepdim
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
                     "Non-constant dims is not support for NormConverter");
    auto keepdim_const = engine->context().get_constant(inputs[3]);
    POROS_CHECK_TRUE((keepdim_const.isBool()), " dims type must be int[] for NormConverter")
    auto keepdim = keepdim_const.toBool();

    // unary_layer
    auto unary_layer = engine->network()->addUnary(*self, nvinfer1::UnaryOperation ::kABS);
    unary_layer->setName((layer_info(node) + "_IUnaryLayer").c_str());
    auto unary_output = unary_layer->getOutput(0);

    // elementwise_layer 1
    auto ew1_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPOW, unary_output, p,
                                     layer_info(node) + "_pow_for_unary");

    POROS_CHECK(ew1_layer, "Unable to create POW layer from node: " << *node);
    auto ew_output = ew1_layer->getOutput(0);

    // reduce_layer
    auto reduce_layer = engine->network()->addReduce(*ew_output, nvinfer1::ReduceOperation::kSUM, dim, keepdim);
    reduce_layer->setName((layer_info(node) + "_IReduceLayer").c_str());
    auto reduce_output = reduce_layer->getOutput(0);

    // elementwise_layer 2
    auto ew2_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPOW, reduce_output, p_inverse,
                                     layer_info(node) + "_pow_for_reduce");

    engine->context().set_tensor(node->outputs()[0], ew2_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << ew2_layer->getOutput(0)->getDimensions();
    return true;
}

bool FrobeniusNormConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    // inputs.size() == 3
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for FrobeniusNormConverter")

    // inputs[0] => self
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for FrobeniusNormConverter is not Tensor as expected");
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node)
    auto self_dims = self->getDimensions();

    // p
    nvinfer1::ITensor *p, *p_inverse;
    float p_scalar = 2;
    p = tensor_to_const(engine, torch::tensor(p_scalar));
    p_inverse = tensor_to_const(engine, torch::tensor(1.0 / p_scalar));

    // inputs[1] => dims
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
                     "Non-constant dims is not support for FrobeniusNormConverter");
    auto dims_const = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((dims_const.isIntList()), " dims type must be int[] for FrobeniusNormConverter")
    auto dims_list = dims_const.toIntList().vec();
    uint32_t dim = 0;
    for (auto d: dims_list) {
        if (d < 0) {
            d = self_dims.nbDims + d;
        }
        dim |= 1 << d;
    }
    // in case of dims_list is empty or invalid, reduce on all axes
    if (dim == 0) {
        dim = pow(2, self_dims.nbDims) - 1;
    }

    // input[2] => keepdim
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
                     "Non-constant dims is not support for FrobeniusNormConverter");
    auto keepdim_const = engine->context().get_constant(inputs[2]);
    POROS_CHECK_TRUE((keepdim_const.isBool()), " dims type must be int[] for FrobeniusNormConverter")
    auto keepdim = keepdim_const.toBool();

    // unary_layer
    auto unary_layer = engine->network()->addUnary(*self, nvinfer1::UnaryOperation ::kABS);
    unary_layer->setName((layer_info(node) + "_IUnaryLayer").c_str());
    auto unary_output = unary_layer->getOutput(0);

    // elementwise_layer 1
    auto ew1_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPOW, unary_output, p,
                                     layer_info(node) + "_pow_for_unary");

    POROS_CHECK(ew1_layer, "Unable to create POW layer from node: " << *node);
    auto ew_output = ew1_layer->getOutput(0);

    // reduce_layer
    auto reduce_layer = engine->network()->addReduce(*ew_output, nvinfer1::ReduceOperation::kSUM, dim, keepdim);
    reduce_layer->setName((layer_info(node) + "_IReduceLayer").c_str());
    auto reduce_output = reduce_layer->getOutput(0);

    // elementwise_layer 2
    auto ew2_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPOW, reduce_output, p_inverse,
                                     layer_info(node) + "_pow_for_reduce");

    engine->context().set_tensor(node->outputs()[0], ew2_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << ew2_layer->getOutput(0)->getDimensions();
    return true;
}

//
POROS_REGISTER_CONVERTER(TensorrtEngine, NormConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, FrobeniusNormConverter);

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
