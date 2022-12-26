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
* @file element_wise.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug 27 15:32:36 CST 2021
* @brief 
**/

#include "poros/converter/gpu/element_wise.h"
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

nvinfer1::ITensor* GreaterOrLessConverter::scalar_to_nvtensor(TensorrtEngine* engine, at::Scalar s) {
    nvinfer1::ITensor* out;
    if (s.isIntegral(false)) {
        auto s_int = s.to<int64_t>();
        auto s_t = torch::tensor({s_int}).to(at::kInt);
        out = tensor_to_const(engine, s_t);
    } else if (s.isBoolean()) {
        auto s_bool = s.to<bool>();
        auto s_t = torch::tensor({s_bool}).to(at::kBool);
        out = tensor_to_const(engine, s_t);
    } else if (s.isFloatingPoint()) {
        auto other_float = s.to<float>();
        auto s_t = torch::tensor({other_float});
        out = tensor_to_const(engine, s_t);
    } else {
        out = nullptr;
        POROS_THROW_ERROR("Unsupported data type for scalar. Found: (" << s.type() << ")");
    }
    return out;
}

/*
"aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
"aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",*/
bool GreaterOrLessConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for GreaterOrLessConverter");
    
    int schema_index = 0;
    for (std::string schema : this->schema_string()) {
        if (node->schema().operator_name() == torch::jit::parseSchema(schema).operator_name()) {
            break;
        }
        schema_index++;
    }

    if (schema_index >= 8 && schema_index <= 11) {
        int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
        int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
        bool output = true;
        if (schema_index == 8) {
            output = (a > b);
        }
        if (schema_index == 9) {
            output = (a < b);
        }
        if (schema_index == 10) {
            output = (a >= b);
        }
        if (schema_index == 11) {
            output = (a <= b);
        }
        engine->context().set_constant(node->outputs()[0], output);
        return true;
    }

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for GreaterOrLessConverter is not Tensor as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto other = engine->context().get_tensor(inputs[1]);
    //when other input is Scalar
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            other = scalar_to_nvtensor(engine, other_const.toScalar());
            if (self->getType() != other->getType()) {
                other = cast_itensor(engine, other, self->getType());
            }
        } else {
            POROS_THROW_ERROR("Unable to get input other value for GreaterOrLessConverter");
        }
    }

    nvinfer1::ElementWiseOperation ew_option;
    std::string name_suffix;
    if (node->kind() == torch::jit::aten::gt || node->kind() == torch::jit::aten::ge) {
        ew_option = nvinfer1::ElementWiseOperation::kGREATER;
        name_suffix = "_greater";
    } else if (node->kind() == torch::jit::aten::lt || node->kind() == torch::jit::aten::le) {
        ew_option = nvinfer1::ElementWiseOperation::kLESS;
        name_suffix = "_less";
    } else {
        POROS_THROW_ERROR("Meet some unknown node kind in GreaterOrLessConverter");
    }

    auto new_layer = add_elementwise(engine,
            ew_option,
            self,
            other,
            layer_info(node) + name_suffix);
    POROS_CHECK(new_layer, "Unable to create element wise layer from node: " << *node);

    //situation: aten::gt or aten::lt
    if (node->kind() == torch::jit::aten::gt || node->kind() == torch::jit::aten::lt) {
        engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
        return true;
    }

    // situation: aten::ge or aten::le, 
    // we should set three layers: kGREATER（or kLESS） and kEQUAL and kOR.
    if (node->kind() == torch::jit::aten::ge || node->kind() == torch::jit::aten::le) {
        //equal layer
        auto equal = add_elementwise(engine,
                nvinfer1::ElementWiseOperation::kEQUAL,
                self,
                other,
                layer_info(node) + "_equal");
        POROS_CHECK(equal, "Unable to create Equal layer from node: " << *node);
        
        //or layer
        auto or_op = engine->network()->addElementWise(
            *new_layer->getOutput(0),
            *equal->getOutput(0),
            nvinfer1::ElementWiseOperation::kOR);
        POROS_CHECK(or_op, "Unable to create Or layer from node: " << *node);

        or_op->setName((layer_info(node) + "_or").c_str());
        engine->context().set_tensor(node->outputs()[0], or_op->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << or_op->getOutput(0)->getDimensions();
        return true;
    }

    POROS_THROW_ERROR("Meet some unknown node kind in GreaterOrLessConverter");
    return false;
}

/*
"aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
"aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",*/
bool EqualOrNotequalConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for EqualOrNotequalConverter");
    
    int schema_index = 0;
    for (std::string schema : this->schema_string()) {
        if (node->schema().operator_name() == torch::jit::parseSchema(schema).operator_name()) {
            break;
        }
        schema_index++;
    }

    if (schema_index == 4 || schema_index == 5) {
        int a = engine->context().get_constant(inputs[0]).toScalar().to<int>();
        int b = engine->context().get_constant(inputs[1]).toScalar().to<int>();
        bool output = true;
        if (schema_index == 4) {
            output = (a == b);
        }
        if (schema_index == 5) {
            output = (a != b);
        }
        engine->context().set_constant(node->outputs()[0], output);
        return true;
    }

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for EqualOrNotequalConverter is not Tensor as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto other = engine->context().get_tensor(inputs[1]);
    //when other input is Scalar
    if (other == nullptr) {
        auto other_const = engine->context().get_constant(inputs[1]);
        if (other_const.isScalar()) {
            auto other_scalar = other_const.toScalar().to<float>();
            other = tensor_to_const(engine, torch::tensor({other_scalar}));
            if (node->kind() == torch::jit::aten::eq) {
                //TODO: when aten::ne situation, we may alse need to cat functions below??
                if (self->getType() == nvinfer1::DataType::kBOOL) {
                    if (other_scalar == 0 || other_scalar == 1) {
                        LOG(INFO) << "Since input tensor is type bool, casting input tensor and scalar to int32";
                        other = cast_itensor(engine, other, nvinfer1::DataType::kINT32);
                        self = cast_itensor(engine, self, nvinfer1::DataType::kINT32);
                    } else {
                        LOG(WARNING) << "Input Tensor has type bool, but scalar is not 0 or 1. Found: " << other_scalar;
                        return false;
                    }
                }
                if (self->getType() != other->getType()) {
                    other = cast_itensor(engine, other, self->getType());
                }
            }
        } else {
            POROS_THROW_ERROR("Unable to get input other value for EqualOrNotequalConverter");
        }
    }

    auto equal_layer = add_elementwise(engine,
            nvinfer1::ElementWiseOperation::kEQUAL,
            self,
            other,
            layer_info(node) + "_equal");
    POROS_CHECK(equal_layer, "Unable to create equal layer from node: " << *node);

    //situation: aten::eq
    if (node->kind() == torch::jit::aten::eq) {
        engine->context().set_tensor(node->outputs()[0], equal_layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << equal_layer->getOutput(0)->getDimensions();
        return true;
    }

    // situation: aten::ne
    // we should set another two layers: all-ones layer and kXOR.
    if (node->kind() == torch::jit::aten::ne) {
        // XOR with ones negates and produces not_equal result
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto ones = at::full({1}, 1, {options});
        auto ones_tensor = tensor_to_const(engine, ones);
        nvinfer1::IIdentityLayer* cast_layer = engine->network()->addIdentity(*ones_tensor);
        cast_layer->setName((layer_info(node) + "_IIdentityLayer").c_str());
        cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);      

        //xor layer
        auto xor_op = add_elementwise(engine,
                nvinfer1::ElementWiseOperation::kXOR,
                cast_layer->getOutput(0),
                equal_layer->getOutput(0),
                layer_info(node) + "_xor");
        POROS_CHECK(xor_op, "Unable to create ne (not equal) layer from node: " << *node);
        engine->context().set_tensor(node->outputs()[0], xor_op->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << xor_op->getOutput(0)->getDimensions();
        return true;
    }

    POROS_THROW_ERROR("Meet some unknown node kind in EqualOrNotequalConverter");
    return false;
}

/*
"aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor",
"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",*/
bool PowOrFloordivideConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for PowOrFloordivideConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for PowOrFloordivideConverter is not Tensor as expected");

    //extract self
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
            POROS_THROW_ERROR("Unable to get input other value for PowOrFloordivideConverter");
        }
    }

    nvinfer1::ElementWiseOperation ew_option;
    std::string name_suffix;
    if (node->kind() == torch::jit::aten::pow) {
        ew_option = nvinfer1::ElementWiseOperation::kPOW;
        name_suffix = "_pow";
    //TODO: handle floor_divide situaition
    // } else if (node->kind() == torch::jit::at::floor_divide) {
    //     ew_option = nvinfer1::ElementWiseOperation::kFLOOR_DIV;
    //     name_suffix = "_floor_div";
    } else {
        POROS_THROW_ERROR("Meet some unknown node kind in PowOrFloordivideConverter");
    }

    auto new_layer = add_elementwise(engine,
            ew_option,
            self,
            other,
            layer_info(node) + name_suffix);
    POROS_CHECK(new_layer, "Unable to create pow or floor_divide layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

/*
"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
"aten::clamp_min(Tensor self, Scalar min) -> Tensor",
"aten::clamp_max(Tensor self, Scalar max) -> Tensor",*/
bool ClampConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2 || inputs.size() == 3), "invaid inputs size for ClampConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[0] for ClampConverter is not Tensor as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto clamp_layer_out = self;

    torch::jit::IValue maybe_min;
    torch::jit::IValue maybe_max;
    if (node->kind() == torch::jit::aten::clamp) {
        maybe_min = engine->context().get_constant(inputs[1]);
        maybe_max = engine->context().get_constant(inputs[2]);
    } else if (node->kind() == torch::jit::aten::clamp_min) {
        maybe_min = engine->context().get_constant(inputs[1]);
        maybe_max = torch::jit::IValue();
    } else { //node->kind() == torch::jit::aten::clamp_max
        maybe_min = torch::jit::IValue();
        maybe_max = engine->context().get_constant(inputs[1]);
    }

    if (maybe_min.isScalar() && maybe_max.isScalar()) {
        // note: same as pytorch, first max, then min
        auto limit = maybe_min.toScalar().to<float>();
        auto limit_tensor = tensor_to_const(engine, torch::tensor({limit}));
        auto limit_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kMAX,
                            self,
                            limit_tensor,
                            layer_info(node) + "_max");
        POROS_CHECK(limit_layer, "Unable to create elementwise(KMAX) layer for node: " << *node);
        clamp_layer_out = limit_layer->getOutput(0);
        limit = maybe_max.toScalar().to<float>();
        limit_tensor = tensor_to_const(engine, torch::tensor({limit}));
        limit_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kMIN,
                            clamp_layer_out,
                            limit_tensor,
                            layer_info(node) + "_min");
        POROS_CHECK(limit_layer, "Unable to create elementwise(KMIN) layer for node: " << *node);
        clamp_layer_out = limit_layer->getOutput(0);
    } else if (maybe_min.isScalar()) {
        auto limit = maybe_min.toScalar().to<float>();
        auto limit_tensor = tensor_to_const(engine, torch::tensor({limit}));
        auto limit_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kMAX,
                            self,
                            limit_tensor,
                            layer_info(node) + "_max");
        POROS_CHECK(limit_layer, "Unable to create elementwise(KMAX) layer for node: " << *node);
        clamp_layer_out = limit_layer->getOutput(0);
    } else if (maybe_max.isScalar()) {
        auto limit = maybe_max.toScalar().to<float>();
        auto limit_tensor = tensor_to_const(engine, torch::tensor({limit}));
        auto limit_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kMIN,
                            self,
                            limit_tensor,
                            layer_info(node) + "_min");
        POROS_CHECK(limit_layer, "Unable to create elementwise(KMIN) layer for node: " << *node);
        clamp_layer_out = limit_layer->getOutput(0);
    }

    engine->context().set_tensor(node->outputs()[0], clamp_layer_out);
    LOG(INFO) << "Output tensor shape: " << clamp_layer_out->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, GreaterOrLessConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, EqualOrNotequalConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, PowOrFloordivideConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ClampConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
