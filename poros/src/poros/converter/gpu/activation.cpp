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
* @file activation.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/activation.h"
#include "poros/converter/gpu/weight.h"
#include "poros/util/macros.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/context/poros_global.h"
#include "poros/util/poros_util.h"
#include "poros/converter/gpu/converter_util.h"

namespace baidu {
namespace mirana {
namespace poros {

bool ActivationConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1 || inputs.size() == 2 
        || inputs.size() == 3 || inputs.size() == 4), 
        "invaid inputs size for ActivationConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ActivationConverter is not Tensor as expected");

    auto nv_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((nv_tensor != nullptr), 
        "Unable to init input tensor for node: " << *node);

    nvinfer1::ActivationType activate_type;
    if (node->kind() == torch::jit::aten::relu || node->kind() == torch::jit::aten::relu_) {
        activate_type = nvinfer1::ActivationType::kRELU;
    } else if (node->kind() == torch::jit::aten::relu6 || node->kind() == torch::jit::aten::relu6_) {
        activate_type = nvinfer1::ActivationType::kRELU;
    } else if (node->kind() == torch::jit::aten::sigmoid || node->kind() == torch::jit::aten::sigmoid_) {
        activate_type = nvinfer1::ActivationType::kSIGMOID;
    } else if (node->kind() == torch::jit::aten::tanh || node->kind() == torch::jit::aten::tanh_) {
        activate_type = nvinfer1::ActivationType::kTANH;
    } else if (node->kind() == torch::jit::aten::leaky_relu) {
        activate_type = nvinfer1::ActivationType::kLEAKY_RELU;
    } else if (node->kind() == torch::jit::aten::hardtanh || node->kind() == torch::jit::aten::hardtanh_) {
        activate_type = nvinfer1::ActivationType::kCLIP;
    } else if (node->kind() == torch::jit::aten::elu) {
        activate_type = nvinfer1::ActivationType::kELU;
    }else if (node->kind() == torch::jit::aten::silu) {
        activate_type = nvinfer1::ActivationType::kSIGMOID;
    }  else {
        POROS_THROW_ERROR("We should never reach here for ActivationConverter, meet Unsupported ActivationType!");
    }

    auto new_layer = engine->network()->addActivation(*nv_tensor, activate_type);

    //set attributes for aten::leaky_relu
    //"aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor",
    if (activate_type == nvinfer1::ActivationType::kLEAKY_RELU) {
        POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for aten::leaky_relu in ActivationConverter");
        auto negative_slopeScalar = (engine->context().get_constant(inputs[1])).toScalar().to<float>();
        new_layer->setAlpha(negative_slopeScalar);
    }

    //set attributes for aten::hardtanh
    //"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
    if (activate_type == nvinfer1::ActivationType::kCLIP) {
        POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for aten::hardtanh in ActivationConverter");
        auto min = (engine->context().get_constant(inputs[1])).toDouble();
        auto max = (engine->context().get_constant(inputs[2])).toDouble();
        new_layer->setAlpha(min);
        new_layer->setBeta(max);
    }

    //set attributes for aten::elu
    //"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"
    if (activate_type == nvinfer1::ActivationType::kELU) {
        POROS_CHECK_TRUE((inputs.size() == 4), "invaid inputs size for aten::hardtanh in ActivationConverter");
        auto alpha = (engine->context().get_constant(inputs[1])).toDouble();
        new_layer->setAlpha(alpha);
    }
    
    new_layer->setName((layer_info(node) + "_IActivationLayer").c_str());
    nvinfer1::ITensor* output = new_layer->getOutput(0);
    if (node->kind() == torch::jit::aten::relu6 || node->kind() == torch::jit::aten::relu6_) {
        nvinfer1::ITensor* relu_output = new_layer->getOutput(0);
        auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(at::kFloat);
        at::Tensor relu6_max = at::tensor({6.0}, options_pyt);
        nvinfer1::ITensor* relu6_max_nv = tensor_to_const(engine, relu6_max);

        auto min_layer = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kMIN,
                            relu_output,
                            relu6_max_nv,
                            layer_info(node) + "_min");
        output = min_layer->getOutput(0);
    }else if (node->kind() == torch::jit::aten::silu) {
        nvinfer1::ITensor* sigmoid_output = new_layer->getOutput(0);
        auto min_layer = add_elementwise(engine,
                                         nvinfer1::ElementWiseOperation::kPROD,
                                         sigmoid_output,
                                         nv_tensor,
                                         layer_info(node) + "_prod");
        output = min_layer->getOutput(0);
    }


    
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output shape: " << output->getDimensions();
    return true;
}

// aten::gelu(Tensor self) -> Tensor
// aten::gelu(Tensor self, *, str approximate='none') -> Tensor
bool GeluActivationConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1 || inputs.size() == 2), "invaid inputs size for GeluActivationConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for GeluActivationConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), 
        "Unable to init input tensor for node: " << *node);
    nvinfer1::DataType type = in->getType();
    
    POROS_CHECK((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF),
        "gelu only supports kFLOAT and kHALF");
    
    std::string pluginName = "CustomGeluPluginDynamic";
    nvinfer1::PluginFieldCollection fc;
    std::vector<nvinfer1::PluginField> f;

    //TODO: maybe need to consider  more about op_precision situation
    // int type_id = ctx->settings.op_precision == nvinfer1::DataType::kFLOAT 
    //         ? 0
    //         : 1; // Integer encoding the DataType (0: FP32, 1: FP16)
    int type_id = (type == nvinfer1::DataType::kFLOAT) ? 0 : 1;
    f.emplace_back(nvinfer1::PluginField("type_id", &type_id, nvinfer1::PluginFieldType::kINT32, 1));

    std::string mode = "gelu";
    f.emplace_back(nvinfer1::PluginField("mode", &mode, nvinfer1::PluginFieldType::kCHAR, 1));

    fc.nbFields = f.size();
    fc.fields = f.data();
    
    auto creator = getPluginRegistry()->getPluginCreator("CustomGeluPluginDynamic", "1", "");
    auto gelu_plugin = creator->createPlugin("gelu", &fc);
    
    POROS_CHECK(gelu_plugin, "Unable to create gelu plugin from TensorRT plugin registry" << *node);
    auto new_layer = 
        engine->network()->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *gelu_plugin);
    new_layer->setName((layer_info(node) + "_plugin_gelu").c_str());
    auto out_tensor = new_layer->getOutput(0);
    engine->context().set_tensor(node->outputs()[0], out_tensor);
    LOG(INFO) << "Output shape: " << out_tensor->getDimensions();
    return true;
}

/*"aten::prelu(Tensor self, Tensor weight) -> Tensor"*/
bool PreluActivationConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for PreluActivationConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for PreluActivationConverter is not Tensor as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    auto maybe_slopes = engine->context().get_constant(inputs[1]);
    POROS_CHECK_TRUE((maybe_slopes.isTensor()), "Unable to init input const-tensor for node: " << *node);
    auto slopes = maybe_slopes.toTensor();  //at::tensor
    //auto slopes_size = sizes_to_nvdim(slopes.sizes());

    //bool to_reshape = false;
    auto original_shape = in->getDimensions();
    
    // Channel dim is the 2nd dim of input. When input has dims < 2, then there is no channel dim and the number of channels = 1.
    at::Tensor weight;
    if (slopes.numel() != 1){
        std::vector<at::Tensor> weights;
        std::vector<int64_t> reshape_shape;
        bool sign = true;
        for (int i = 0; i < original_shape.nbDims; i++) {
            if (original_shape.d[i] == slopes.numel() && sign) {
                sign = false;
                continue;
            }
            if (!sign) {
                reshape_shape.push_back(original_shape.d[i]);
            }
        }

        for (int64_t i = 0; i < slopes.numel(); i++) {
            auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
            auto tmp = at::ones(reshape_shape, options_pyt);
            weights.push_back((slopes[i] * tmp).unsqueeze(0));
        }

        weight = torch::cat(weights, 0);
        weight = weight.unsqueeze(0);
    } else {
        weight = slopes;
    } 

    auto slope_tensor = tensor_to_const(engine, weight);
    auto new_layer = engine->network()->addParametricReLU(*in, *slope_tensor);
    new_layer->setName((layer_info(node) + "_IParametricReLULayer").c_str());
    auto out_tensor = new_layer->getOutput(0);

    engine->context().set_tensor(node->outputs()[0], out_tensor);
    LOG(INFO) << "Output shape: " << out_tensor->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ActivationConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, GeluActivationConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, PreluActivationConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
