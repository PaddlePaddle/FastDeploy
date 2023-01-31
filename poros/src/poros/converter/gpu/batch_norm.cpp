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
* @file batch_norm.cpp
* @author tianjinjin@baidu.com
* @date Sun Aug 15 22:23:03 CST 2021
* @brief 
**/

#include "poros/converter/gpu/batch_norm.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/weight.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

/*
aten::batch_norm(Tensor input, 
Tensor? weight, 
Tensor? bias, 
Tensor? running_mean, 
Tensor? running_var, 
bool training, 
float momentum, 
float eps, 
bool cudnn_enabled) -> Tensor
*/
bool BatchNormConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    
    POROS_CHECK_TRUE((inputs.size() == 9), "invaid inputs size for BatchNormConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for BatchNormConverter is not Tensor as expected");
    // weight & bias & running_mean & running_var
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for BatchNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for BatchNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for BatchNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[4]->node()->kind() == torch::jit::prim::Constant),
        "input[4] for BatchNormConverter is not come from prim::Constant as expected");

    auto input = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((input != nullptr), 
        "Unable to init input tensor for node: " << *node);
    auto orig_shape = input->getDimensions();
    auto shape = nvdim_to_sizes(orig_shape);
    auto tensor_type = nvtype_to_attype(input->getType());
    auto options = torch::TensorOptions().dtype(tensor_type);

    torch::Tensor gamma, beta, mean, var;
    auto maybe_gamma = engine->context().get_constant(inputs[1]);
    auto maybe_beta = engine->context().get_constant(inputs[2]);
    auto maybe_mean = engine->context().get_constant(inputs[3]);
    auto maybe_bar = engine->context().get_constant(inputs[4]);

    if (maybe_gamma.isTensor()) {
        gamma = maybe_gamma.toTensor();
    } else {
        gamma = at::full({shape}, 1, {options});
    }

    if (maybe_beta.isTensor()) {
        beta = maybe_beta.toTensor();
    } else {
        beta = at::full({shape}, 1, {options});
    }

    if (maybe_mean.isTensor()) {
        mean = maybe_mean.toTensor();
    } else {
        mean = at::full({shape}, 0, {options});
    }

    if (maybe_bar.isTensor()) {
        var = maybe_bar.toTensor();
    } else {
       var = at::full({shape}, 0, {options}); 
    }

    auto eps = engine->context().get_constant(inputs[7]).to<float>();

    // Expand spatial dims from 1D to 2D if needed
    bool expandDims = (orig_shape.nbDims < 4);
    if (expandDims) {
        input = add_padding(engine, node, input, 4);
    }

    auto scale = gamma / torch::sqrt(var + eps);
    auto bias = beta - mean * scale;

    auto scale_weights = Weights(scale);
    auto bias_weights = Weights(bias);

    auto power = Weights(at::ones_like(scale));
    auto bn = engine->network()->addScaleNd(
        *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
    bn->setName((layer_info(node) + "_IScaleLayer").c_str());
    // Un-pad bn output if needed
    auto out_tensor = add_unpadding(engine, node, bn->getOutput(0), orig_shape.nbDims);
    engine->context().set_tensor(node->outputs()[0], out_tensor);
    LOG(INFO) << "Output tensor shape: " << out_tensor->getDimensions();
    return true;
}

/*
aten::instance_norm(Tensor input,
Tensor? weight,
Tensor? bias,
Tensor? running_mean,
Tensor? running_var,
bool use_input_stats,
float momentum,
float eps,
bool cudnn_enabled) -> Tensor
*/
bool InstanceNormConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    
    POROS_CHECK_TRUE((inputs.size() == 9), "invaid inputs size for InstanceNormConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for InstanceNormConverter is not Tensor as expected");
    // weight & bias & running_mean & running_var
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for InstanceNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for InstanceNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for InstanceNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[4]->node()->kind() == torch::jit::prim::Constant),
        "input[4] for InstanceNormConverter is not come from prim::Constant as expected");

    auto input = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((input != nullptr), 
        "Unable to init input tensor for node: " << *node);
    auto orig_shape = input->getDimensions();
    auto shape = nvdim_to_sizes(orig_shape);
    auto tensor_type = nvtype_to_attype(input->getType());
    auto options = torch::TensorOptions().dtype(tensor_type);

    // Expand spatial dims from 1D to 2D if needed
    bool expand_dims = (orig_shape.nbDims < 4);
    if (expand_dims) {
        input = add_padding(engine, node, input, 4);
    }

    torch::Tensor weight, bias, mean, var;
    auto maybe_weight = engine->context().get_constant(inputs[1]);
    auto maybe_bias = engine->context().get_constant(inputs[2]);
    auto maybe_mean = engine->context().get_constant(inputs[3]);
    auto maybe_var = engine->context().get_constant(inputs[4]);

    if (maybe_weight.isTensor()) {
        weight = maybe_weight.toTensor().cpu().contiguous();
    } else {
        weight = at::ones(shape[1], options).cpu().contiguous();
    }

    if (maybe_bias.isTensor()) {
        bias = maybe_bias.toTensor().cpu().contiguous();
    } else {
        bias = at::zeros(shape[1], options).cpu().contiguous();
    }

    auto eps = static_cast<float>(engine->context().get_constant(inputs[7]).toDouble());

    //TODO: 确认此处设置 ”或“ 还是 “与” 合适
    if (maybe_mean.isTensor() && maybe_var.isTensor()) {
        mean = maybe_mean.toTensor();
        var = maybe_var.toTensor();
        
        auto scale = weight.to(mean.options()) / torch::sqrt(var + eps);
        auto new_bias = bias.to(mean.options()) - mean * scale;

        auto scale_weights = Weights(scale);
        auto bias_weights = Weights(new_bias);
        
        auto power = Weights(at::ones_like(scale));
        auto bn = engine->network()->addScaleNd(
            *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
        bn->setName((layer_info(node) + "_IScaleLayer").c_str());
        // Un-pad bn output if needed
        auto out_tensor = add_unpadding(engine, node, bn->getOutput(0), orig_shape.nbDims);
        engine->context().set_tensor(node->outputs()[0], out_tensor);
        LOG(INFO) << "Output tensor shape: " << out_tensor->getDimensions();
        return true;
    }

    // https://github.com/NVIDIA/TensorRT/tree/release/8.4/plugin/instanceNormalizationPlugin
    
    const int relu = 0;
    const float alpha = 0;
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back(nvinfer1::PluginField("epsilon", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
    f.emplace_back(nvinfer1::PluginField(
        "scales", weight.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, weight.numel()));
    f.emplace_back(nvinfer1::PluginField(
        "bias", bias.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, bias.numel()));
    f.emplace_back(nvinfer1::PluginField("relu", &relu, nvinfer1::PluginFieldType::kINT32, 1));
    f.emplace_back(nvinfer1::PluginField("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1));
    
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    
    auto creator = getPluginRegistry()->getPluginCreator("InstanceNormalization_TRT", "1", "");
    auto instance_norm_plugin = creator->createPlugin("instance_norm", &fc);
    
    POROS_CHECK(instance_norm_plugin, "Unable to create instance_norm plugin from TensorRT plugin registry" << *node);
    auto new_layer = engine->network()->addPluginV2(
        reinterpret_cast<nvinfer1::ITensor* const*>(&input), 1, *instance_norm_plugin);
    new_layer->setName((layer_info(node) + "_plugin_instance_norm").c_str());
    nvinfer1::ITensor* output = new_layer->getOutput(0);

    if (expand_dims) {
        output = add_unpadding(engine, node, output, orig_shape.nbDims);
    }

    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, BatchNormConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, InstanceNormConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
