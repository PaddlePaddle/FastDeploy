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
* @file add.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/add.h"
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

/**
 * @brief unify type : According to the type promotion rules of torch, 
 *        when one of self or other is float type, the other also becomes float.
 * @param [in] engine : trt engine
 * @param [in] node : current node
 * @param [in] self : self ITensor
 * @param [in] other : other ITensor
 * @return
**/
static void unify_type(TensorrtEngine* engine, 
                    const torch::jit::Node *node, 
                    nvinfer1::ITensor*& self, 
                    nvinfer1::ITensor*& other) {
    if (self->getType() == nvinfer1::DataType::kFLOAT && 
        other->getType() == nvinfer1::DataType::kINT32) {
        auto id_layer = engine->network()->addIdentity(*other);
        id_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        id_layer->setName((layer_info(node) + "_IIdentityLayer_other_to_float").c_str());
        other = id_layer->getOutput(0);
    }

    if (other->getType() == nvinfer1::DataType::kFLOAT && 
        self->getType() == nvinfer1::DataType::kINT32) {
        auto id_layer = engine->network()->addIdentity(*self);
        id_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        id_layer->setName((layer_info(node) + "_IIdentityLayer_self_to_float").c_str());
        self = id_layer->getOutput(0);
    }
}

/*
"aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"*/
bool AddConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    
    // aten::add.int(int a, int b) -> (int)
    // aten::add.t(t[] a, t[] b) -> (t[])
    if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[4]).operator_name() ||
    node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[5]).operator_name()) {
        POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for AddConverter");
        if (check_inputs_tensor_scalar(engine, node)) {
            // 获取int对应的nvtensor
            nvinfer1::ITensor* a = this->get_tensor_scalar(inputs[0]);
            nvinfer1::ITensor* b = this->get_tensor_scalar(inputs[1]);
            // 判断是否为空 (get_constant失败时可能为空)
            // 为空时返回false, 让子图fallback
            POROS_CHECK_TRUE((a != nullptr && b != nullptr), 
                                node_info(node) + std::string("get nvtensor type int false."));
            if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[4]).operator_name()) {
                // a和b相加并返回
                nvinfer1::ILayer* add_layer = add_elementwise(engine, 
                                                nvinfer1::ElementWiseOperation::kSUM, 
                                                a, b, layer_info(node) + "_sum");
                POROS_CHECK(add_layer, "Unable to create add layer from node: " << *node);
                nvinfer1::ITensor* output = add_layer->getOutput(0);
                engine->context().set_tensor(node->outputs()[0], output);
                LOG(INFO) << "Output tensor shape: " << output->getDimensions();
            } else {
                std::vector<nvinfer1::ITensor*> inputs_nvtensor;
                // 将所有int对应的nvtensor加入vector, 最后cat起来
                inputs_nvtensor.push_back(a);
                inputs_nvtensor.push_back(b);
                nvinfer1::IConcatenationLayer* concat_layer = 
                        engine->network()->addConcatenation(inputs_nvtensor.data(), inputs_nvtensor.size());
                concat_layer->setAxis(0);
                concat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
                engine->context().set_tensor(node->outputs()[0], concat_layer->getOutput(0));
            }            
        } else {
            torch::jit::IValue a_ivalue = engine->context().get_constant(inputs[0]);
            if (a_ivalue.isInt()) {
                int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
                int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
                engine->context().set_constant(node->outputs()[0], a + b);
            } else if (a_ivalue.isIntList()) {
                std::vector<int64_t> a_vec = engine->context().get_constant(inputs[0]).toIntList().vec();
                std::vector<int64_t> b_vec = engine->context().get_constant(inputs[1]).toIntList().vec();
                a_vec.insert(a_vec.end(), b_vec.begin(), b_vec.end());
                auto output_ivalue = c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(a_vec)));
                engine->context().set_constant(node->outputs()[0], output_ivalue);   
            } else {
                // a and b are tensorlists
                if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors())) {
                    std::vector<nvinfer1::ITensor*> in_tensor_a, in_tensor_b;
                    engine->context().get_tensorlist(inputs[0], in_tensor_a);
                    engine->context().get_tensorlist(inputs[1], in_tensor_b);
                    in_tensor_a.insert(in_tensor_a.end(), in_tensor_b.begin(), in_tensor_b.end());
                    engine->context().set_tensorlist(node->outputs()[0], in_tensor_a);
                } else {
                    LOG(INFO) << *node->maybeSchema() << " meets unkown input type.";
                    return false;
                }
            }
        }
        return true;
    }

    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for AddConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for AddConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for AddConverter is not come from prim::Constant as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    //extract scalar
    //TODO: check input scalar is int type situation
    auto scalar_ivalue = (engine->context().get_constant(inputs[2]));
    // 先转成float去接input[2]输入
    auto scalar = scalar_ivalue.toScalar().to<float>();
    //which one is better?
    //auto scalar = ((engine->context().get_constant(inputs[2]))->to<int>();
    //auto scalar = ((engine->context().get_constant(inputs[2])).to<c10::Scalar>()).to<float>();

    //extract other
    auto other = engine->context().get_tensor(inputs[1]);
    //situation1: ---------- when other input is Scalar -------------
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            // 先转成float去接input[1]输入
            auto other_scalar = other_const.toScalar().to<float>();
            at::Tensor prod_tensor = torch::tensor({other_scalar * scalar});
            // input[1]和input[2]若本身是int，相乘结果需转成int
            if (scalar_ivalue.isInt() && other_const.isInt()) {
                prod_tensor = prod_tensor.to(at::ScalarType::Int);
            }
            other = tensor_to_const(engine, prod_tensor);
        } else {
            POROS_THROW_ERROR("Unable to get input other value for AddConverter");
        }
    //situation2:  ---------- when other input is Tensor -------------
    } else {
        const double EPSILON = 1e-9;
        if (fabs(scalar - 1.0) > EPSILON) {
            nvinfer1::ITensor* alphaTensor = nullptr;
            // input[1]和input[2]若本身是int，则input[2]需转回int。否则trt中float和int相乘为0。
            if (scalar_ivalue.isInt() && other->getType() == nvinfer1::DataType::kINT32) {
                alphaTensor = tensor_to_const(engine, torch::tensor({scalar}).to(at::ScalarType::Int));
            } else {
                alphaTensor = tensor_to_const(engine, torch::tensor({scalar}));
            }
            auto scaleLayer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kPROD,
                            other,
                            alphaTensor,
                            layer_info(node) + "_prod");
            POROS_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *node);
            other = scaleLayer->getOutput(0);
        }
    }

    unify_type(engine, node, self, other);

    auto add = add_elementwise(engine, 
            nvinfer1::ElementWiseOperation::kSUM, 
            self, 
            other,
            layer_info(node) + "_sum");
    POROS_CHECK(add, "Unable to create add layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], add->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << add->getOutput(0)->getDimensions();
    return true;
}

/*
"aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
"aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",*/
bool SubConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    // aten::sub.int(int a, int b) -> (int)
    if (node->schema().operator_name() == torch::jit::parseSchema(this->schema_string()[4]).operator_name()) {
        POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for SubConverter");
        if (check_inputs_tensor_scalar(engine, node)) {
            // 获取int对应的nvtensor
            nvinfer1::ITensor* a = this->get_tensor_scalar(inputs[0]);
            nvinfer1::ITensor* b = this->get_tensor_scalar(inputs[1]);
            // 判断是否为空 (get_constant失败时可能为空)
            // 为空时返回false, 让子图fallback
            POROS_CHECK_TRUE((a != nullptr && b != nullptr), 
                                node_info(node) + std::string("get int nvtensor false."));
            // a和b相加并返回
            nvinfer1::ILayer* sub_layer = add_elementwise(engine, 
                                            nvinfer1::ElementWiseOperation::kSUB, 
                                            a, b, layer_info(node) + "_sub");
            POROS_CHECK(sub_layer, "Unable to create sub layer from node: " << *node);
            nvinfer1::ITensor* output = sub_layer->getOutput(0);
            engine->context().set_tensor(node->outputs()[0], output);
            LOG(INFO) << "Output tensor shape: " << output->getDimensions();
        } else {
            int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
            int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
            engine->context().set_constant(node->outputs()[0], a - b);
        }
        return true;
    }

    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for SubConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SubConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for SubConverter is not come from prim::Constant as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    //extract scalar
    //TODO: check input scalar is int type situation
    auto scalar_ivalue = (engine->context().get_constant(inputs[2]));
    // 先转成float去接input[2]输入
    auto scalar = scalar_ivalue.toScalar().to<float>();

    //extract other
    auto other = engine->context().get_tensor(inputs[1]);
    //situation1: ---------- when other input is Scalar -------------
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            // 先转成float去接input[1]输入
            auto other_scalar = other_const.toScalar().to<float>();
            at::Tensor prod_tensor = torch::tensor({other_scalar * scalar});
            // input[1]和input[2]若本身是int，相乘结果需转成int
            if (scalar_ivalue.isInt() && other_const.isInt()) {
                prod_tensor = prod_tensor.to(at::ScalarType::Int);
            }
            other = tensor_to_const(engine, prod_tensor);
        } else {
            POROS_THROW_ERROR("Unable to get input other value for MulConverter");
        }
    //situation2:  ---------- when other input is Tensor -------------
    } else {
        const double EPSILON = 1e-9;
        if (fabs(scalar - 1.0) > EPSILON) {
            nvinfer1::ITensor* alphaTensor = nullptr;
            // input[1]和input[2]若本身是int，则input[2]需转回int。否则trt中float和int相乘为0。
            if (scalar_ivalue.isInt() && other->getType() == nvinfer1::DataType::kINT32) {
                alphaTensor = tensor_to_const(engine, torch::tensor({scalar}).to(at::ScalarType::Int));
            } else {
                alphaTensor = tensor_to_const(engine, torch::tensor({scalar}));
            }
            auto scaleLayer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kPROD,
                            other,
                            alphaTensor,
                            layer_info(node) + "_prod");
            POROS_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *node);
            other = scaleLayer->getOutput(0);
        }
    }

    unify_type(engine, node, self, other);

    auto sub = add_elementwise(engine, 
            nvinfer1::ElementWiseOperation::kSUB, 
            self,
            other,
            layer_info(node) + "_sub");
    POROS_CHECK(sub, "Unable to create sub layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], sub->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << sub->getOutput(0)->getDimensions();
    return true;
}

// aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)
// aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)
bool RsubConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for SubConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SubConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for SubConverter is not come from prim::Constant as expected");

    //extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    //extract scalar
    //TODO: check input scalar is int type situation
    auto scalar_ivalue = engine->context().get_constant(inputs[2]);
    // 先转成float去接input[2]输入
    auto scalar = scalar_ivalue.toScalar().to<float>();

    // self * alpha
    const double EPSILON = 1e-9;
    if (fabs(scalar - 1.0) > EPSILON) {
        nvinfer1::ITensor* alpha_tensor = nullptr;
        // input[0]和input[2]若本身是int，则input[2]需转回int。否则trt中float和int相乘为0。
        if (scalar_ivalue.isInt() && self->getType() == nvinfer1::DataType::kINT32) {
            alpha_tensor = tensor_to_const(engine, torch::tensor({scalar}).to(at::ScalarType::Int));
        } else {
            alpha_tensor = tensor_to_const(engine, torch::tensor({scalar}));
        }
        auto scaleLayer = add_elementwise(engine,
                        nvinfer1::ElementWiseOperation::kPROD,
                        self,
                        alpha_tensor,
                        layer_info(node) + "_prod");
        POROS_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *node);
        self = scaleLayer->getOutput(0);
    }

    //extract other
    auto other = engine->context().get_tensor(inputs[1]);
    //situation1: ---------- when other input is Scalar -------------
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            // 先转成float去接input[1]输入
            auto other_scalar = other_const.toScalar().to<float>();
            at::Tensor other_tensor = torch::tensor({other_scalar});
            if (other_const.isInt() && self->getType() == nvinfer1::DataType::kINT32) {
                other_tensor = other_tensor.to(at::ScalarType::Int);
            }
            other = tensor_to_const(engine, other_tensor);
        } else {
            POROS_THROW_ERROR("Unable to get input other value for MulConverter");
        }
    } 

    unify_type(engine, node, self, other);

    auto sub = add_elementwise(engine, 
            nvinfer1::ElementWiseOperation::kSUB, 
            other,
            self,
            layer_info(node) + "_rsub");
    POROS_CHECK(sub, "Unable to create sub layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], sub->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << sub->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, AddConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, SubConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, RsubConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu