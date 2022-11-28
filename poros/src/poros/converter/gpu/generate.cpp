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
* @file generate.cpp
* @author tianshaoqing@baidu.com
* @date Mon Dec 6 14:29:20 CST 2021
* @brief 
**/

#include "poros/converter/gpu/generate.h"
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

// aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
bool ZerosLikeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6), "invaid inputs size for ZerosLikeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ZerosLikeConverter is not Tensor as expected");
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[1]);
    if (maybe_type.isNone()) {
        nvinfer1::ILayer* sub_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kSUB, 
                                                        self, self, layer_info(node) + "_sub");
        nvinfer1::ITensor* output = sub_layer->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();
        return true;
    } else {
        nvinfer1::ITensor* shape_tensor = engine->network()->addShape(*self)->getOutput(0);
        int32_t self_rank = (shape_tensor->getDimensions()).d[0];

        at::ScalarType input_type = maybe_type.toScalarType();
        nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                        nvinfer1::FillOperation::kLINSPACE);
        fill_layer->setInput(0, *shape_tensor);  // 设置output shape

        at::Tensor value_tensor = torch::tensor(0).to(input_type);
        nvinfer1::ITensor* value_itensor = tensor_to_const(engine, value_tensor);
        fill_layer->setInput(1, *value_itensor); // 初始值

        at::Tensor delta_tensor = torch::zeros(self_rank).to(input_type); // 每个方向上的变化，所以self_rank个0
        nvinfer1::ITensor* delta_itensor = tensor_to_const(engine, delta_tensor);
        fill_layer->setInput(2, *delta_itensor);
        fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());

        nvinfer1::ITensor* output = fill_layer->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();
        return true;
    }
}

// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"
bool ZerosConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for ZerosConverter");

    bool has_tensor_scalar = false;
    has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    nvinfer1::ITensor* output = nullptr;
    // extract dtype
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[1]);

    if (has_tensor_scalar) {
        // extract size
        nvinfer1::ITensor* shape_tensor = engine->context().get_tensor(inputs[0]); // from size
        POROS_CHECK_TRUE((shape_tensor != nullptr), "Unable to init input tensor for node: " << *node);

        nvinfer1::Dims self_dims = shape_tensor->getDimensions();
        int64_t self_rank = self_dims.d[0];

        nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                        nvinfer1::FillOperation::kLINSPACE);
        fill_layer->setInput(0, *shape_tensor);  // 设置output shape
        // default type is float
        at::Tensor value_tensor = torch::tensor(0.0, torch::kFloat32);
        at::Tensor delta_tensor = torch::zeros(self_rank, torch::kFloat32); // 每个方向上的变化，所以self_rank个0
        // type conversion
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
            delta_tensor = delta_tensor.to(maybe_type.toScalarType());
        }
        nvinfer1::ITensor* value_itensor = tensor_to_const(engine, value_tensor);
        fill_layer->setInput(1, *value_itensor); // 初始值
        nvinfer1::ITensor* delta_itensor = tensor_to_const(engine, delta_tensor);
        fill_layer->setInput(2, *delta_itensor);
        fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
        output = fill_layer->getOutput(0);
    } else {
        std::vector<int64_t> self_vec = (engine->context().get_constant(inputs[0])).toIntList().vec();
        at::Tensor value_tensor = torch::zeros(self_vec, torch::kFloat32);
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
        }
        output = tensor_to_const(engine, value_tensor);
    }
    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}


// aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
bool OnesConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5), "invaid inputs size for OnesConverter");

    bool has_tensor_scalar = false;
    has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    nvinfer1::ITensor* output = nullptr;
    // extract dtype
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[1]);

    if (has_tensor_scalar) {
        // extract size
        nvinfer1::ITensor* shape_tensor = engine->context().get_tensor(inputs[0]); // from size
        POROS_CHECK_TRUE((shape_tensor != nullptr), "Unable to init input tensor for node: " << *node);

        nvinfer1::Dims self_dims = shape_tensor->getDimensions();
        int64_t self_rank = self_dims.d[0];

        nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                        nvinfer1::FillOperation::kLINSPACE);
        fill_layer->setInput(0, *shape_tensor);  // 设置output shape
        // default type is float
        at::Tensor value_tensor = torch::tensor(1.0, torch::kFloat32);
        at::Tensor delta_tensor = torch::zeros(self_rank, torch::kFloat32); // 每个方向上的变化，所以self_rank个0
        // type conversion
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
            delta_tensor = delta_tensor.to(maybe_type.toScalarType());
        }
        nvinfer1::ITensor* value_itensor = tensor_to_const(engine, value_tensor);
        fill_layer->setInput(1, *value_itensor); // 初始值
        nvinfer1::ITensor* delta_itensor = tensor_to_const(engine, delta_tensor);
        fill_layer->setInput(2, *delta_itensor);
        fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
        output = fill_layer->getOutput(0);
    } else {
        std::vector<int64_t> self_vec = (engine->context().get_constant(inputs[0])).toIntList().vec();
        at::Tensor value_tensor = torch::ones(self_vec, torch::kFloat32);
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
        }
        output = tensor_to_const(engine, value_tensor);
    }
    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}


// aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
bool FullConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6), "invaid inputs size for FullConverter");

    bool has_tensor_scalar = false;
    has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    nvinfer1::ITensor* output = nullptr;
    // extract fill_value
    torch::jit::IValue maybe_value = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((!maybe_value.isNone()), "Unable to init input fill value for node: " << *node);
    float fill_value = maybe_value.toScalar().toFloat();
    // extract dtype
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[2]);

    if (has_tensor_scalar) {
        // extract size
        nvinfer1::ITensor* shape_tensor = engine->context().get_tensor(inputs[0]); // from size
        POROS_CHECK_TRUE((shape_tensor != nullptr), "Unable to init input tensor for node: " << *node);

        nvinfer1::Dims self_dims = shape_tensor->getDimensions();
        int64_t self_rank = self_dims.d[0];

        nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                        nvinfer1::FillOperation::kLINSPACE);
        fill_layer->setInput(0, *shape_tensor);  // 设置output shape
        // default type is float
        at::Tensor value_tensor = torch::tensor(fill_value, torch::kFloat32);
        at::Tensor delta_tensor = torch::zeros(self_rank, torch::kFloat32); // 每个方向上的变化，所以self_rank个0
        // type conversion
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
            delta_tensor = delta_tensor.to(maybe_type.toScalarType());
        }
        nvinfer1::ITensor* value_itensor = tensor_to_const(engine, value_tensor);
        fill_layer->setInput(1, *value_itensor); // 初始值
        nvinfer1::ITensor* delta_itensor = tensor_to_const(engine, delta_tensor);
        fill_layer->setInput(2, *delta_itensor);
        fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
        output = fill_layer->getOutput(0);
    } else {
        std::vector<int64_t> self_vec = (engine->context().get_constant(inputs[0])).toIntList().vec();
        at::Tensor value_tensor = torch::ones(self_vec, torch::kFloat32) * fill_value;
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
        }
        output = tensor_to_const(engine, value_tensor);
    }
    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

// reduce input_tensor with shape
static nvinfer1::ITensor* reduce_dim1_to_dim0(TensorrtEngine* engine, nvinfer1::ITensor* input_tensor) {
    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    if (input_dims.nbDims == 1 && input_dims.d[0] == 1) {
        nvinfer1::IShuffleLayer* shuffle_l = engine->network()->addShuffle(*input_tensor);
        nvinfer1::Dims squeeze_dim;
        squeeze_dim.nbDims = 0;
        shuffle_l->setReshapeDimensions(squeeze_dim);
        return shuffle_l->getOutput(0);
    } else {
        return input_tensor;
    }
}

// aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
// aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
bool ArangeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 5 || inputs.size() == 6), "invaid inputs size for ArangeConverter.");
    // start、end目前只支持int类型
    POROS_CHECK_TRUE((node->inputs()[0]->type()->kind() == c10::TypeKind::IntType), 
                                        "The type of input[0] for ArangeConverter must be Int.");
    if (inputs.size() == 6) {
        POROS_CHECK_TRUE((node->inputs()[1]->type()->kind() == c10::TypeKind::IntType), 
                                        "The type of input[1] for ArangeConverter must be Int.");
    }

    int type_input_index = inputs.size() - 4;
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[type_input_index]);

    if (check_inputs_tensor_scalar(engine, node)) {
        nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                        nvinfer1::FillOperation::kLINSPACE);
        if (inputs.size() == 5) {
            nvinfer1::ITensor* end_tensor = this->get_tensor_scalar(inputs[0]);
            // 设置output shape
            fill_layer->setInput(0, *end_tensor);  
            // 设置 start 和 delta
            at::Tensor value_tensor = torch::tensor(0, torch::kInt32);
            at::Tensor delta_tensor = torch::ones(1, torch::kInt32);
            auto value_itensor = tensor_to_const(engine, value_tensor);
            fill_layer->setInput(1, *value_itensor);
            auto delta_itensor = tensor_to_const(engine, delta_tensor); 
            fill_layer->setInput(2, *delta_itensor);
        } else {
            nvinfer1::ITensor* start_tensor = this->get_tensor_scalar(inputs[0]);
            nvinfer1::ITensor* end_tensor = this->get_tensor_scalar(inputs[1]);
            // arange_size = end - start
            nvinfer1::ITensor* arange_size = add_elementwise(engine,
                                                nvinfer1::ElementWiseOperation::kSUB,
                                                end_tensor,
                                                start_tensor,
                                                layer_info(node) + "_get_arange_size")->getOutput(0);
            // 设置output shape
            fill_layer->setInput(0, *arange_size);
            // 设置 start 和 delta
            start_tensor = reduce_dim1_to_dim0(engine, start_tensor);
            fill_layer->setInput(1, *start_tensor);
            at::Tensor delta_tensor = torch::ones(1, torch::kInt32);
            auto delta_itensor = tensor_to_const(engine, delta_tensor);
            fill_layer->setInput(2, *delta_itensor);
        }
        fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
        nvinfer1::ITensor* output = fill_layer->getOutput(0);

        if (!maybe_type.isNone()) {
            at::ScalarType scalar_type = maybe_type.toScalarType();
            if (scalar_type == at::ScalarType::Long) {
                scalar_type = at::ScalarType::Int;
                LOG(WARNING) << "aten::arange Converter meets c10::ScalarType::Long tensor type, change this to c10::ScalarType::Int. "
                    << "Attention: this may leed to percision change";
            }
            nvinfer1::DataType output_type = attype_to_nvtype(scalar_type);
            // Set datatype for data to dtype
            auto identity = engine->network()->addIdentity(*output);
            identity->setName((layer_info(node) + "_identity_output").c_str());
            identity->setOutputType(0, output_type);
            output = identity->getOutput(0);
        }
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();

    } else {
        at::Tensor value_tensor;
        if (inputs.size() == 5) {
            int64_t end = engine->context().get_constant(inputs[0]).toInt();
            value_tensor = torch::arange(end, torch::kInt);

        } else {
            int64_t start = engine->context().get_constant(inputs[0]).toInt();
            int64_t end = engine->context().get_constant(inputs[1]).toInt();
            value_tensor = torch::arange(start, end, torch::kInt);
        } 
        if (!maybe_type.isNone()) {
            value_tensor = value_tensor.to(maybe_type.toScalarType());
        }
        nvinfer1::ITensor* output = tensor_to_const(engine, value_tensor);
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    }
    return true;
}

// aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)
bool TensorConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for TensorConverter");
    // extract dtype
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[1]);

    nvinfer1::ITensor* output = nullptr;
    if (check_inputs_tensor_scalar(engine, node)) {
        output = this->get_tensor_scalar(inputs[0]);
        if (!maybe_type.isNone()) {
            at::ScalarType scalar_type = maybe_type.toScalarType();
            if (scalar_type == at::ScalarType::Long) {
                scalar_type = at::ScalarType::Int;
                LOG(WARNING) << "aten::tensor Converter meets c10::ScalarType::Long tensor type, change this to c10::ScalarType::Int. "
                    << "Attention: this may leed to percision change";
            }
            auto output_type = attype_to_nvtype(scalar_type);
            // Set datatype for data to dtype
            auto identity = engine->network()->addIdentity(*output);
            identity->setName((layer_info(node) + "_IIdentityLayer_for_output").c_str());
            identity->setOutputType(0, output_type);
            output = identity->getOutput(0);
        }
        // mark: 06.30 by tsq
        // 如果schema为aten::tensor.int，当其输出为子图输出时，即它的output会输出到torchscript，那么需要变换output rank为0。否则会出core。
        // 理论上来说应该所有aten::tensor.int输出的tensor rank都为0，但这样输出output->getDimensions()为空[]
        // 暂时不清楚rank为0的nvtensor给其他op会有什么影响，所以先限制aten::tensor输出为子图输出时才squeeze
        bool need_squeeze_dim = false;
        if (node->hasUses()) {
            auto users_list = node->output(0)->uses();
            for (size_t i = 0; i < users_list.size(); i++) {
                if (users_list[i].user->kind() == torch::jit::prim::Return) {
                    need_squeeze_dim = true;
                    break;
                }
            }
        }

        if (need_squeeze_dim && inputs[0]->type()->kind() == c10::TypeKind::IntType) {
            nvinfer1::IShuffleLayer* shuffle_l = engine->network()->addShuffle(*output);
            nvinfer1::Dims squeeze_dim;
            squeeze_dim.nbDims = 0;
            shuffle_l->setReshapeDimensions(squeeze_dim);
            shuffle_l->setName((layer_info(node) + "_IShuffleLayer").c_str());
            output = shuffle_l->getOutput(0);
            engine->context().set_tensor(node->outputs()[0], output);
            return true;
        }

    } else {
        // extract dtype
        at::Tensor input_data = engine->context().get_constant(inputs[0]).toTensor();
        if (!maybe_type.isNone()) {
            input_data = input_data.to(maybe_type.toScalarType());
        }
        output = tensor_to_const(engine, input_data);
    }
    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

// aten::linspace(Scalar start, Scalar end, int? steps=None, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
bool LinspaceConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
        at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 7), "invaid inputs size for LinspaceConverter.");

    bool has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
    
    auto fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
    if (has_tensor_scalar) {
        nvinfer1::ITensor* start = this->get_tensor_scalar(inputs[0]);
        nvinfer1::ITensor* end = this->get_tensor_scalar(inputs[1]);
        nvinfer1::ITensor* step = this->get_tensor_scalar(inputs[2]);
        // steps为None时默认值为100
        if (step == nullptr) {
            step = tensor_to_const(engine, at::tensor({100}, at::ScalarType::Float));
        }
        // 默认输出类型为float，以下操作也需要转为float
        // alpha=start，delta=(end - start) / (step - 1)
        if (start->getType() != nvinfer1::DataType::kFLOAT) {
            auto identity = engine->network()->addIdentity(*start);
            identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
            identity->setName((layer_info(node) + "_IIdentityLayer_for_start").c_str());
            start = identity->getOutput(0);
        }
        if (end->getType() != nvinfer1::DataType::kFLOAT) {
            auto identity = engine->network()->addIdentity(*end);
            identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
            identity->setName((layer_info(node) + "_IIdentityLayer_for_end").c_str());
            end = identity->getOutput(0);
        }
        if (step->getType() != nvinfer1::DataType::kFLOAT) {
            auto identity = engine->network()->addIdentity(*step);
            identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
            identity->setName((layer_info(node) + "_IIdentityLayer_for_step").c_str());
            step = identity->getOutput(0);
        }
        // (end - start)
        nvinfer1::ILayer* sub_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kSUB, 
                                                        end, start, layer_info(node) + "_sub(end_start)");
        nvinfer1::ITensor* length = sub_layer->getOutput(0);
        // (step - 1)
        nvinfer1::ITensor* one = tensor_to_const(engine, at::tensor({1}, at::ScalarType::Float));
        nvinfer1::ILayer* sub_layer2 = add_elementwise(engine, nvinfer1::ElementWiseOperation::kSUB, 
                                                        step, one, layer_info(node) + "_sub(step_one)");
        nvinfer1::ITensor* step_sub_one = sub_layer2->getOutput(0);
        // (end - start) / (step - 1)
        nvinfer1::ILayer* div_layer = add_elementwise(engine, nvinfer1::ElementWiseOperation::kDIV, 
                                                        length, step_sub_one, layer_info(node) + "_div(get_delta)");
        nvinfer1::ITensor* delta = div_layer->getOutput(0);
        // step需要转回int32作为Ifilllayer input0的输入，用于指定输出的dim
        if (step->getType() == nvinfer1::DataType::kFLOAT) {
            auto identity = engine->network()->addIdentity(*step);
            identity->setOutputType(0, nvinfer1::DataType::kINT32);
            identity->setName((layer_info(node) + "_IIdentityLayer_for_step_back").c_str());
            step = identity->getOutput(0);
        }
        // 输出只有一维，Ifilllayer需要start的rank为0（check_inputs_tensor_scalar中start scalar转nvtensor时自带了1维）
        if (start->getDimensions().nbDims > 0) {
            nvinfer1::IShuffleLayer* shuffle_l = engine->network()->addShuffle(*start);
            nvinfer1::Dims start_dim;
            start_dim.nbDims = 0;
            shuffle_l->setReshapeDimensions(start_dim);
            shuffle_l->setName((layer_info(node) + "_IShuffleLayer_for_start").c_str());
            start = shuffle_l->getOutput(0);
        }
        fill_layer->setInput(0, *step);
        fill_layer->setInput(1, *start);
        fill_layer->setInput(2, *delta);
    } else {
        torch::jit::IValue start_ivalue = engine->context().get_constant(inputs[0]);
        torch::jit::IValue end_ivalue = engine->context().get_constant(inputs[1]);
        torch::jit::IValue maybe_step = engine->context().get_constant(inputs[2]);
        float start = start_ivalue.toScalar().to<float>();
        float end = end_ivalue.toScalar().to<float>();
        float step = 100.0;
        if (!maybe_step.isNone()) {
            step = maybe_step.toScalar().to<float>();
        }
        float delta = (end - start) / (step - 1);
        std::vector<int64_t> output_dims = {(int64_t)step};
        fill_layer->setDimensions(sizes_to_nvdim(output_dims));
        fill_layer->setAlpha(start);
        fill_layer->setBeta(delta);
    }

    fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
    nvinfer1::ITensor* output = fill_layer->getOutput(0);

    // extract dtype
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[3]);
    // 如果输出不为空，则最后变换输出类型
    if (!maybe_type.isNone()) {
        nvinfer1::DataType output_type = attype_to_nvtype(maybe_type.toScalarType());
        auto identity = engine->network()->addIdentity(*output);
        identity->setName((layer_info(node) + "_IIdentityLayer_for_output").c_str());
        identity->setOutputType(0, output_type);
        output = identity->getOutput(0);
    }
    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

// aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
bool FulllikeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 7), "invaid inputs size for FulllikeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for FulllikeConverter is not Tensor as expected");
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    nvinfer1::Dims self_dims = self->getDimensions();
    
    // 先转成float去接input[1]输入
    auto scalar_ivalue = (engine->context().get_constant(inputs[1]));
    float scalar = scalar_ivalue.toScalar().to<float>();

    // extract type
    torch::jit::IValue maybe_type = engine->context().get_constant(inputs[2]);

    bool is_dynamic = check_nvtensor_is_dynamic(self);

    nvinfer1::IFillLayer* fill_layer = engine->network()->addFill(nvinfer1::Dims{1, {1}}, 
                                                                    nvinfer1::FillOperation::kLINSPACE);
    // set fill shape
    if (is_dynamic) {
        nvinfer1::ITensor* shape_tensor = engine->network()->addShape(*self)->getOutput(0);
        fill_layer->setInput(0, *shape_tensor);
    } else {
        fill_layer->setDimensions(self_dims);
    }

    at::ScalarType init_type = (inputs[0]->type()->cast<c10::TensorType>())->scalarType().value();
    if (init_type == at::ScalarType::Long) {
        init_type = at::ScalarType::Int;
    } else if (init_type == at::ScalarType::Double) {
        init_type = at::ScalarType::Float;
    }
    // 默认输出类型和self一致，与torch保持一致
    at::Tensor value_tensor = torch::tensor(scalar, {init_type});
    at::Tensor delta_tensor = torch::zeros(self_dims.nbDims, {init_type}); // 每个方向上的变化，所以self_rank个0
    if (!maybe_type.isNone()) {
        at::ScalarType input_type = maybe_type.toScalarType();
        if (input_type == at::ScalarType::Long) {
            input_type = at::ScalarType::Int;
        } else if (input_type == at::ScalarType::Double) {
            input_type = at::ScalarType::Float;
        }
        value_tensor = value_tensor.to(input_type);
        delta_tensor = delta_tensor.to(input_type);
    }

    nvinfer1::ITensor* value_itensor = tensor_to_const(engine, value_tensor);
    fill_layer->setInput(1, *value_itensor); // 初始值

    nvinfer1::ITensor* delta_itensor = tensor_to_const(engine, delta_tensor);
    fill_layer->setInput(2, *delta_itensor);
    nvinfer1::ITensor* output = fill_layer->getOutput(0);

    fill_layer->setName((layer_info(node) + "_IFillLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ZerosLikeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ZerosConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, OnesConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, FullConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ArangeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, TensorConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, LinspaceConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, FulllikeConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
