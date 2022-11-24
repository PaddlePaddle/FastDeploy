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
* @file to.cpp
* @author wangrui39@baidu.com
* @date Saturday November 13 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/to.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/engine_context.h"
#include "poros/util/macros.h"
#include "poros/context/poros_global.h"
#include "poros/converter/gpu/weight.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

static void long_to_int(at::ScalarType &scalar_type) {
    if (scalar_type == at::kLong && PorosGlobalContext::instance().get_poros_options().long_to_int == true) {
        scalar_type = at::kInt;
        LOG(WARNING) << "gen_tensor_type meets at::KLong tensor type, change this to at::KInt. "
                    << "Attention: this may leed to percision change";
    }
}

bool ToConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5 || inputs.size() == 6 || 
                inputs.size() == 8), "invaid inputs size for ToConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())),
        "inputs[0] for ToConverter is not Tensor as expected");
    auto self = engine->context().get_tensor(inputs[0]);
    nvinfer1::DataType output_type = self->getType();

    // aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
    if (inputs[1]->type()->isSubtypeOf(c10::TensorType::get())) {
        auto other = engine->context().get_tensor(inputs[1]);
        output_type = other->getType();

    // aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
    // aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))
    } else if (inputs[2]->type()->str() == "int" && inputs[3]->type()->str() == "bool") {
        auto scalar_type = engine->context().get_constant(inputs[2]).toScalarType();
        long_to_int(scalar_type);
        output_type = attype_to_nvtype(scalar_type);
        if (engine->context().get_constant(inputs[1]).toDevice().is_cuda()) {
            auto device = nvinfer1::TensorLocation::kDEVICE;
            self->setLocation(device);
        } else {
            LOG(WARNING) << "Set tensor device to HOST but only DEVICE is supported";
            return false;
        }

    /*  aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
        aten::to.dtype_layout(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, 
                                                            bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)*/
    } else if (inputs[1]->type()->str() == "int") {
        auto scalar_type = engine->context().get_constant(inputs[1]).toScalarType();
        long_to_int(scalar_type);
        output_type = attype_to_nvtype(scalar_type);
    // Input err
    } else {
        POROS_THROW_ERROR("Meet some unsupported inputs value type in ToConstructConverter" << *node);
        return false;
    }

    // Set datatype for self to dtype
    // 注：尽管此处output_type可能和input_type一样，但保险起见也需要过一下identity_layer，否则execute_engine时可能发生错误。
    // 例如：aten::to的input和output tensor同时被mark成engine的输出，如果不走identity_layer，那么这两个tensor其实是一个tensor。
    // build_engine时trt会报 xxx has been marked as output（trt不支持重复标记输出，只会覆盖之前的输出。）
    // 原本期望输出两个实际却只有一个，execute_engine获取输出时会出core。
    // todo: 同类型转换的aten::to.dtype也可以在graph中干掉
    auto identity = engine->network()->addIdentity(*self);
    identity->setOutputType(0, output_type);
    identity->setName((layer_info(node) + "_IIdentityLayer_for_self").c_str());
    self = identity->getOutput(0);
    // setOutputType可能不起作用，用setType再次确保self的类型发生了转换
    self->setType(output_type);

    engine->context().set_tensor(node->outputs()[0], self);
    LOG(INFO) << "Output tensor shape: " << self->getDimensions();
    return true;
}

// prim::NumToTensor.Scalar(Scalar a) -> (Tensor)
bool NumtotensorConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for NumtotensorConverter");
    // 如果是tensor封装的scalar直接向后传
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    if (self != nullptr) {
        engine->context().set_tensor(node->outputs()[0], self);
        LOG(INFO) << "Output tensor shape: " << self->getDimensions();
    } else {
        // 如果传入的是真实的scalar
        auto input_scalar = engine->context().get_constant(inputs[0]);
        if (!input_scalar.isScalar()) {
            POROS_THROW_ERROR("prim::NumToTensor input[0] is not scalar!");
            return false;
        }
        nvinfer1::ITensor* output_tensor = nullptr;
        if (input_scalar.isInt()) {
            output_tensor = tensor_to_const(engine, at::tensor(input_scalar.toInt(), torch::kInt));
        } else if (input_scalar.isDouble()) {
            output_tensor = tensor_to_const(engine, at::tensor(input_scalar.toDouble(), torch::kDouble).to(at::ScalarType::Float));
        } else if (input_scalar.isBool()) {
            output_tensor = tensor_to_const(engine, at::tensor(input_scalar.toBool(), torch::kBool));
        } else {
            POROS_THROW_ERROR("prim::NumToTensor Converter meets an unsupported scalar type, which leads to fail.");
            return false;
        }
        engine->context().set_tensor(node->outputs()[0], output_tensor);
        LOG(INFO) << "Output tensor shape: " << output_tensor->getDimensions();
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ToConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, NumtotensorConverter);

} // baidu
} // mirana
} // poros
