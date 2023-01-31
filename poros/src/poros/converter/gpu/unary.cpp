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
* @file unary.cpp
* @author tianjinjin@baidu.com
* @date Mon Sep  6 20:23:14 CST 2021
* @brief 
**/

#include "poros/converter/gpu/unary.h"
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
"aten::cos(Tensor self) -> Tensor",
*/
bool UnaryConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for UnaryConverter");
    if (node->schema().operator_name() != 
        torch::jit::parseSchema("aten::floor.float(float a) -> (int)").operator_name()) {
        POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
            "input[0] for UnaryConverter is not Tensor as expected");
    }

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    nvinfer1::UnaryOperation trt_type;
    switch (node->kind()) {
        case torch::jit::aten::cos:
            trt_type = nvinfer1::UnaryOperation::kCOS;
            break;
        case torch::jit::aten::acos:
            trt_type = nvinfer1::UnaryOperation::kACOS;
            break;
        case torch::jit::aten::cosh:
            trt_type = nvinfer1::UnaryOperation::kCOSH;
            break;
        case torch::jit::aten::sin:
            trt_type = nvinfer1::UnaryOperation::kSIN;
            break;
        case torch::jit::aten::asin:
            trt_type = nvinfer1::UnaryOperation::kASIN;
            break;
        case torch::jit::aten::sinh:
            trt_type = nvinfer1::UnaryOperation::kSINH;
            break;
        case torch::jit::aten::tan:
            trt_type = nvinfer1::UnaryOperation::kTAN;
            break;
        case torch::jit::aten::atan:
            trt_type = nvinfer1::UnaryOperation::kATAN;
            break;
        case torch::jit::aten::abs:
            trt_type = nvinfer1::UnaryOperation::kABS;
            break;
        case torch::jit::aten::floor:
            trt_type = nvinfer1::UnaryOperation::kFLOOR;
            break;
        case torch::jit::aten::reciprocal:
            trt_type = nvinfer1::UnaryOperation::kRECIP;
            break;
        case torch::jit::aten::log:
            trt_type = nvinfer1::UnaryOperation::kLOG;
            break;
        case torch::jit::aten::ceil:
            trt_type = nvinfer1::UnaryOperation::kCEIL;
            break;
        case torch::jit::aten::sqrt:
            trt_type = nvinfer1::UnaryOperation::kSQRT;
            break;
        case torch::jit::aten::exp:
            trt_type = nvinfer1::UnaryOperation::kEXP;
            break;
        case torch::jit::aten::neg:
            trt_type = nvinfer1::UnaryOperation::kNEG;
            break;
        case torch::jit::aten::erf:
            trt_type = nvinfer1::UnaryOperation::kERF;
            break;
        case torch::jit::aten::asinh:
            trt_type = nvinfer1::UnaryOperation::kASINH;
            break;
        case torch::jit::aten::acosh:
            trt_type = nvinfer1::UnaryOperation::kACOSH;
            break;
        case torch::jit::aten::atanh:
            trt_type = nvinfer1::UnaryOperation::kATANH;
            break;
        case torch::jit::aten::log2:
            trt_type = nvinfer1::UnaryOperation::kLOG;
            break;
        case torch::jit::aten::log10:
            trt_type = nvinfer1::UnaryOperation::kLOG;
            break;
        case torch::jit::aten::round:
            trt_type = nvinfer1::UnaryOperation::kROUND;
            break;
        default:
            POROS_THROW_ERROR("We should never reach here for UnaryConverter, meet Unsupported node kind!");
    }
    //IUnaryLayer only support: operation NEG not allowed on type Int32
    nvinfer1::DataType self_type = self->getType();
    const nvinfer1::DataType allowed_type = trt_type == nvinfer1::UnaryOperation::kNOT ? nvinfer1::DataType::kBOOL : nvinfer1::DataType::kFLOAT;
    bool should_cast = self_type == allowed_type ? false : true;
    if (should_cast) {
        nvinfer1::IIdentityLayer* cast_layer = engine->network()->addIdentity(*self);
        cast_layer->setName((layer_info(node) + "_IIdentityLayer").c_str());
        cast_layer->setOutputType(0, allowed_type);
        self = cast_layer->getOutput(0);  
    }

    auto unary = engine->network()->addUnary(*self, trt_type); 
    POROS_CHECK(unary, "Unable to create unary layer from node: " << *node);
    unary->setName((layer_info(node) + "_IUnaryLayer").c_str());
    auto output = unary->getOutput(0);
    if (trt_type == nvinfer1::UnaryOperation::kLOG) {
        nvinfer1::ITensor* alphaTensor = nullptr; 
        if (node->kind() == torch::jit::aten::log2) {
            alphaTensor = tensor_to_const(engine, torch::tensor(std::log2(std::exp(1)), {torch::kFloat32}));
        } else if (node->kind() == torch::jit::aten::log10) {
            alphaTensor = tensor_to_const(engine, torch::tensor(std::log10(std::exp(1)), {torch::kFloat32}));
        } else {
            // need not to do anything.
        }
        // ln(x) * log2(e) = log2(x)
        // ln(x) * log10(e) = log10(x)
        if (alphaTensor != nullptr) {
            auto scaleLayer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kPROD,
                            output,
                            alphaTensor,
                            layer_info(node) + std::string("_prod"));
            POROS_CHECK(scaleLayer, "Unable to create scale layer from node: " << *node);
            output = scaleLayer->getOutput(0);
        }
    }
    if (node->schema().operator_name() == 
        torch::jit::parseSchema("aten::floor.float(float a) -> (int)").operator_name()) {
        auto identity = engine->network()->addIdentity(*output);
        identity->setOutputType(0, nvinfer1::DataType::kINT32);
        identity->setName((layer_info(node) + "_IIdentityLayer_for_output").c_str());
        output = identity->getOutput(0);
    } else if (should_cast) {
        nvinfer1::IIdentityLayer* castback_layer = engine->network()->addIdentity(*output);
        castback_layer->setName((layer_info(node) + "_IIdentityLayer_for_output").c_str());
        castback_layer->setOutputType(0, self_type);
        output = castback_layer->getOutput(0);
    }
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, UnaryConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
