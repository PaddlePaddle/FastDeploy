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
* @file mul_div.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/mul_div.h"
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
aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
aten::mul.Scalar(Tensor self, Scalar other) -> Tensor*/
bool MulConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for MulConverter");

    // 先判断schema是否是 aten::mul.int(int a, int b) -> (int)
    if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[4]).operator_name()) {
        // 检查int是否以nvtensor的形式输入
        if (check_inputs_tensor_scalar(engine, node)) {
            // 获取int对应的nvtensor
            nvinfer1::ITensor *a = this->get_tensor_scalar(inputs[0]);
            nvinfer1::ITensor *b = this->get_tensor_scalar(inputs[1]);
            // 判断是否为空 (get_constant失败时可能为空)
            // 为空时返回false, 让子图fallback
            POROS_CHECK_TRUE((a != nullptr && b != nullptr),
                             node_info(node) + std::string("get int nvtensor false."));
            // a和b相乘并返回
            nvinfer1::ILayer *mul_layer = add_elementwise(engine,
                                                          nvinfer1::ElementWiseOperation::kPROD,
                                                          a, b, layer_info(node) + "_prod");
            POROS_CHECK(mul_layer, "Unable to create mul layer from node: " << *node);
            nvinfer1::ITensor *output = mul_layer->getOutput(0);
            engine->context().set_tensor(node->outputs()[0], output);
            LOG(INFO) << "Output tensor shape: " << output->getDimensions();
            return true;
        } else {
            int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
            int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
            engine->context().set_constant(node->outputs()[0], a * b);
            return true;
        }
    }

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for MulConverter is not Tensor as expected");

    // Should implement self * other
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto other = engine->context().get_tensor(inputs[1]);
    // when other input is Scalar
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            auto other_scalar = other_const.toScalar().to<float>();
            other = tensor_to_const(engine, torch::tensor({other_scalar}));
        } else {
            POROS_THROW_ERROR("Unable to get input other value for MulConverter");
        }
    }
    // 遇到过aten::mul(float tensor, int scalar)的输入情况，都转成float就行
    if (self->getType() == nvinfer1::DataType::kFLOAT && other->getType() == nvinfer1::DataType::kINT32) {
        nvinfer1::IIdentityLayer* identity_layer = engine->network()->addIdentity(*other);
        identity_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        identity_layer->setName((layer_info(node) + "_IIdentityLayer_for_other").c_str());
        other = identity_layer->getOutput(0);
    } else if (other->getType() == nvinfer1::DataType::kFLOAT && self->getType() == nvinfer1::DataType::kINT32) {
        nvinfer1::IIdentityLayer* identity_layer = engine->network()->addIdentity(*self);
        identity_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        identity_layer->setName((layer_info(node) + "_IIdentityLayer_for_self").c_str());
        self = identity_layer->getOutput(0);
    }

    auto mul = add_elementwise(engine, nvinfer1::ElementWiseOperation::kPROD, self, other, layer_info(node) + "_prod");
    POROS_CHECK(mul, "Unable to create mul layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], mul->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << mul->getOutput(0)->getDimensions();
    return true;
}

/*
aten::div.Tensor(Tensor self, Tensor other) -> Tensor
aten::div.Scalar(Tensor self, Scalar other) -> Tensor*/
bool DivConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for DivConverter");

    // aten::div.int(int a, int b) -> (float)
    if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[4]).operator_name() || \
        node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[5]).operator_name()) {  
        if (check_inputs_tensor_scalar(engine, node)) {
            nvinfer1::ITensor *a = this->get_tensor_scalar(inputs[0]);
            nvinfer1::ITensor *b = this->get_tensor_scalar(inputs[1]);
            POROS_CHECK_TRUE((a != nullptr && b != nullptr),
                             node_info(node) + std::string("get int nvtensor false."));
            // Set datatype for input tensor to kFLOAT
            auto identity_layer1 = engine->network()->addIdentity(*a);
            identity_layer1->setOutputType(0, nvinfer1::DataType::kFLOAT);
            identity_layer1->setName((layer_info(node) + "_IIdentityLayer_for_input0").c_str());
            nvinfer1::ITensor *a_float = identity_layer1->getOutput(0);
            // Set datatype for input tensor to kFLOAT
            auto identity_layer2 = engine->network()->addIdentity(*b);
            identity_layer2->setOutputType(0, nvinfer1::DataType::kFLOAT);
            identity_layer2->setName((layer_info(node) + "_IIdentityLayer_for_input1").c_str());
            nvinfer1::ITensor *b_float = identity_layer2->getOutput(0);

            nvinfer1::ILayer *div_layer = add_elementwise(engine,
                                                          nvinfer1::ElementWiseOperation::kDIV,
                                                          a_float, b_float, layer_info(node) + "_div");
            POROS_CHECK(div_layer, "Unable to create div layer from node: " << *node);
            nvinfer1::ITensor *output = div_layer->getOutput(0);
            LOG(INFO) << "div output type: " << output->getType();
            engine->context().set_tensor(node->outputs()[0], output);
            LOG(INFO) << "Output tensor shape: " << output->getDimensions();
            return true;
        } else {
            int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
            int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
            float output = float(a) / float(b);
            engine->context().set_constant(node->outputs()[0], output);
            return true;
        }
    }

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for DivConverter is not Tensor as expected");

    // Should implement self / other
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto other = engine->context().get_tensor(inputs[1]);
    //when other input is Scalar
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            auto other_scalar = other_const.toScalar().to<float>();
            other = tensor_to_const(engine, torch::tensor({other_scalar}));
        } else {
            POROS_THROW_ERROR("Unable to get input other value for DivConverter");
        }
    }

    if (self->getType() == nvinfer1::DataType::kINT32) {
        nvinfer1::IIdentityLayer* identity_self_layer = engine->network()->addIdentity(*self);
        identity_self_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        self = identity_self_layer->getOutput(0);
    }

    if (other->getType() == nvinfer1::DataType::kINT32) {
        nvinfer1::IIdentityLayer* identity_other_layer = engine->network()->addIdentity(*other);
        identity_other_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        other = identity_other_layer->getOutput(0);
    }

    auto div = add_elementwise(engine, nvinfer1::ElementWiseOperation::kDIV, self, other, layer_info(node) + "_div");
    POROS_CHECK(div, "Unable to create div layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], div->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << div->getOutput(0)->getDimensions();
    return true;
}

// aten::floordiv.int(int a, int b) -> (int)
// aten::__round_to_zero_floordiv(int a, int b) -> (int)
bool FloordivConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for ScalarFloordivConverter");

    // 输入是nvtensor
    if (check_inputs_tensor_scalar(engine, node)) {
        // __round_to_zero_div 待支持
        nvinfer1::ITensor *a = this->get_tensor_scalar(inputs[0]);
        nvinfer1::ITensor *b = this->get_tensor_scalar(inputs[1]);

        POROS_CHECK_TRUE((a != nullptr && b != nullptr),
                         node_info(node) + std::string("get int nvtensor false."));

        nvinfer1::ElementWiseOperation opreation;
        std::string nv_layer_name;
        if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[1]).operator_name()) {
            opreation = nvinfer1::ElementWiseOperation::kDIV;
            nv_layer_name = layer_info(node) + "_div";
        } else {
            opreation = nvinfer1::ElementWiseOperation::kFLOOR_DIV;
            nv_layer_name = layer_info(node) + "_floor_div";
        }

        nvinfer1::ILayer *floordiv_layer = add_elementwise(engine,
                                                           opreation,
                                                           a, b, nv_layer_name);
        POROS_CHECK(floordiv_layer, "Unable to create floordiv layer from node: " << *node);
        nvinfer1::ITensor *output = floordiv_layer->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    } else {
        // 输入是int ivalue
        int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
        int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
        POROS_CHECK_TRUE((b != 0), "invaid inputs[1] for ScalarFloordivConverter, which is equal to zero");
        int output = 0;
        if (node->kind() == node_kind()[0]) {
            output = std::floor(float(a) / float(b));
        } else {
            output = int(a / b);
        }
        engine->context().set_constant(node->outputs()[0], output);
    }
    return true;
}

//aten::remainder.Scalar(Tensor self, Scalar other) -> (Tensor)
//aten::remainder.Tensor(Tensor self, Tensor other) -> (Tensor)
bool RemainderConverter::converter(TensorrtEngine *engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value *> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for RemainderConverter");

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
                     "input[0] for RemainderConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->type()->kind() == c10::TypeKind::FloatType ||
                      inputs[1]->type()->kind() == c10::TypeKind::IntType ||
                      inputs[1]->type()->kind() == c10::TypeKind::TensorType),
                     "input[1] for RemainderConverter is not Scalar as expected");

    nvinfer1::ITensor *self = engine->context().get_tensor(inputs[0]);

    nvinfer1::ITensor *other;

    if (inputs[1]->type()->kind() == c10::TypeKind::TensorType) {
        other = engine->context().get_tensor(inputs[1]);
    } else {
        other = tensor_to_const(engine,
                                torch::tensor(engine->context().get_constant(inputs[1]).toDouble(), torch::kFloat32));
    }

    POROS_CHECK_TRUE((self != nullptr && other != nullptr),
                     node_info(node) + std::string("get int nvtensor false."));

    // floor_div
    nvinfer1::ILayer *floordiv_layer = add_elementwise(engine,
                                                       nvinfer1::ElementWiseOperation::kFLOOR_DIV,
                                                       self, other, layer_info(node) + "_floor_div");
    POROS_CHECK(floordiv_layer, "Unable to create floordiv layer from node: " << *node);
    nvinfer1::ITensor *floordiv_output = floordiv_layer->getOutput(0);

    // prod
    nvinfer1::ILayer *prod_layer = add_elementwise(engine,
                                                   nvinfer1::ElementWiseOperation::kPROD,
                                                   floordiv_output, other, layer_info(node) + "_prod");
    POROS_CHECK(prod_layer, "Unable to create prod layer from node: " << *node);
    nvinfer1::ITensor *prod_output = prod_layer->getOutput(0);

    // sub
    nvinfer1::ILayer *sub_layer = add_elementwise(engine,
                                                  nvinfer1::ElementWiseOperation::kSUB,
                                                  self, prod_output, layer_info(node) + "_sub");
    POROS_CHECK(sub_layer, "Unable to create sub layer from node: " << *node);
    nvinfer1::ITensor *output = sub_layer->getOutput(0);

    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, MulConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, DivConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, FloordivConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, RemainderConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
