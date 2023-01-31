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
* @file group_norm.cpp
* @author tianshaoqing@baidu.com
* @date Fri Jan 21 15:28:37 CST 2022
* @brief
**/

#include "poros/converter/gpu/group_norm.h"
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

/**
 * @brief expand_gamma_beta
 * such as: 
 * input shape is [2, 10, 3, 3].
 * This function first shuffle gamma(or beta) shape from [10] to [10, 1, 1],
 * then slice gamma(or beta) shape from [10, 1, 1] to [10, 3, 3].
 *
 * @param [in] engine : engine of group_norm converter.
 * @param [in] weight_tensor : gamma or beta tensor.
 * @param [in] weight_shuffle_dims : gamma or beta shuffle dims.
 * @param [in] target_size : if input is dynamic, this parameter determines the slice size.
 * @param [in] target_dims : if input is not dynamic, this parameter determines the slice size.
 * @param [in] is_dynamic : input is dynamic or not.
 * @return nvinfer1::ITensor*
 * @retval 
**/
static nvinfer1::ITensor* expand_gamma_beta(TensorrtEngine* engine,
                                            nvinfer1::ITensor* weight_tensor, 
                                            const nvinfer1::Dims& weight_shuffle_dims,
                                            nvinfer1::ITensor* target_size, 
                                            const nvinfer1::Dims& target_dims,
                                            const bool& is_dynamic,
                                            const std::string& name) {
    nvinfer1::IShuffleLayer* shuffle_l = engine->network()->addShuffle(*weight_tensor);
    shuffle_l->setReshapeDimensions(weight_shuffle_dims);
    std::vector<int64_t> start(target_dims.nbDims, 0), stride(target_dims.nbDims, 0);
    stride[0] = 1;
    nvinfer1::ISliceLayer* slice_l = engine->network()->addSlice(*(shuffle_l->getOutput(0)), 
                                                                    sizes_to_nvdim(start), 
                                                                    target_dims, 
                                                                    sizes_to_nvdim(stride));
    if (is_dynamic) {
        slice_l->setInput(2, *target_size);
    }
    slice_l->setName(name.c_str());
    return slice_l->getOutput(0);
}

// aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
bool GroupNormConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6), "invaid inputs size for GroupNormConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for GroupNormConverter is not Tensor as expected");
    // weight & bias
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for GroupNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for GroupNormConverter is not come from prim::Constant as expected");

    nvinfer1::ITensor* input = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((input != nullptr), "Unable to init input tensor for node: " << *node);

    //extract rank of input and check it
    int input_rank = input->getDimensions().nbDims;
    if (input_rank < 2) {
        LOG(WARNING) << *node << ": num of input dimensions must be greater than 2, but got " << input_rank;
        return false;
    }

    //extract tensor type info of input
    at::ScalarType tensor_type = nvtype_to_attype(input->getType());
    auto options = torch::TensorOptions().dtype(tensor_type);
    
    //extract shape of input
    nvinfer1::Dims ori_shape = input->getDimensions();
    nvinfer1::ITensor* ori_shape_tensor = engine->network()->addShape(*input)->getOutput(0);
    std::vector<int64_t> ori_shape_vec = nvdim_to_sizes(ori_shape);
    int64_t channel_size = ori_shape_vec[1];

    //extract num_groups info
    int64_t num_groups = (engine->context().get_constant(inputs[1])).toInt();

    // check input is dynamic or not
    bool is_dynamic = check_nvtensor_is_dynamic(input);
    
    if (!is_dynamic && channel_size % num_groups != 0) {
        LOG(WARNING) << *node << ":Expected number of channels in input to be divisible by num_groups," 
        << " but got input of shape " << ori_shape << ", and num_groups=" << num_groups;
        return false;
    }
    
    // ATTENTION! we need to static_cast eps from double to float. otherwise coredump happened in instancenorm plugin
    //double eps = (engine->context().get_constant(inputs[4])).toDouble();
    float eps = static_cast<float>(engine->context().get_constant(inputs[4]).toDouble());

    //reshape input
    std::vector<int64_t> new_shape = {0, num_groups, -1};
    nvinfer1::ITensor* new_shape_tensor = tensor_to_const(engine, torch::tensor(new_shape, torch::kInt64));    
    nvinfer1::IShuffleLayer* input_shuffle = engine->network()->addShuffle(*input);
    input_shuffle->setInput(1, *new_shape_tensor);
    input_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_input").c_str());
    nvinfer1::ITensor*  input_reshaped = input_shuffle->getOutput(0);

    // const std::vector<int32_t> expand_axes{3};
    // input_reshaped = unsqueeze_itensor(engine, input_reshaped, expand_axes);
    nvinfer1::ITensor* norm_input = add_padding(engine, node, input_reshaped, 4);

    torch::Tensor weight_ = at::ones(num_groups, options).cpu().contiguous();
    torch::Tensor bias_ = at::zeros(num_groups, options).cpu().contiguous();

    //set to instancenorm first
    const int relu = 0;
    const float alpha = 0;
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back(nvinfer1::PluginField("epsilon", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
    f.emplace_back(nvinfer1::PluginField("scales", weight_.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, weight_.numel()));
    f.emplace_back(nvinfer1::PluginField("bias", bias_.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, bias_.numel()));
    f.emplace_back(nvinfer1::PluginField("relu", &relu, nvinfer1::PluginFieldType::kINT32, 1));
    f.emplace_back(nvinfer1::PluginField("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1));
    
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    
    auto creator = getPluginRegistry()->getPluginCreator("InstanceNormalization_TRT", "1", "");
    auto instance_norm_plugin = creator->createPlugin("instance_norm", &fc);
    
    POROS_CHECK(instance_norm_plugin, "Unable to create instance_norm plugin from TensorRT plugin registry" << *node);
    auto new_layer = engine->network()->addPluginV2(
        reinterpret_cast<nvinfer1::ITensor* const*>(&norm_input), 1, *instance_norm_plugin);
    new_layer->setName((layer_info(node) + "_plugin_instance_norm").c_str());
    nvinfer1::ITensor* norm_reshaped = new_layer->getOutput(0);

    nvinfer1::IShuffleLayer* norm_shuffle = engine->network()->addShuffle(*norm_reshaped);
    norm_shuffle->setInput(1, *ori_shape_tensor);
    norm_shuffle->setName((layer_info(node) + "_IShuffleLayer_for_input_back").c_str());
    nvinfer1::ITensor* norm = norm_shuffle->getOutput(0);

    std::vector<int> axes(input_rank - 2);
    std::iota(axes.begin(), axes.end(), 1);

    nvinfer1::ITensor* weight = engine->context().get_tensor(inputs[2]);
    if (weight == nullptr) {
        weight = tensor_to_const(engine, at::ones(1, options));
    }
    weight =  unsqueeze_itensor(engine, weight, axes);

    nvinfer1::ITensor* bias = engine->context().get_tensor(inputs[3]);
    if (bias == nullptr) {
        bias = tensor_to_const(engine, at::zeros(1, options));
    }
    bias =  unsqueeze_itensor(engine, bias, axes);

    //add(mul(norm, weight), bias)
    nvinfer1::ITensor* mul_tensor = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kPROD,
                            norm,
                            weight,
                            layer_info(node) + "_prod")->getOutput(0);
    
    nvinfer1::ITensor* final_tensor = add_elementwise(engine,
                            nvinfer1::ElementWiseOperation::kSUM,
                            mul_tensor,
                            bias,
                            layer_info(node) + "_sum")->getOutput(0);

    engine->context().set_tensor(node->outputs()[0], final_tensor);
    LOG(INFO) << "Output tensor shape: " << final_tensor->getDimensions();
    return true;
}

// aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
bool GroupNormConverter::converter_old(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6), "invaid inputs size for GroupNormConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for GroupNormConverter is not Tensor as expected");
    // weight & bias
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for GroupNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for GroupNormConverter is not come from prim::Constant as expected");

    nvinfer1::ITensor* input = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((input != nullptr), "Unable to init input tensor for node: " << *node);
    nvinfer1::Dims ori_shape = input->getDimensions();
    
    if (ori_shape.nbDims < 2) {
        LOG(WARNING) << *node << ": num of input dimensions must be greater than 2, but got " << ori_shape.nbDims;
        return false;
    }
    
    std::vector<int64_t> ori_shape_vec = nvdim_to_sizes(ori_shape);
    int64_t num_groups = (engine->context().get_constant(inputs[1])).toInt();

    // check input is dynamic or not
    bool is_dynamic = check_nvtensor_is_dynamic(input);
    
    if (!is_dynamic && ori_shape_vec[1] % num_groups != 0) {
        LOG(WARNING) << *node << ":Expected number of channels in input to be divisible by num_groups," 
        << " but got input of shape " << ori_shape << ", and num_groups=" << num_groups;
        return false;
    }

    // Unwrap eps.
    double eps = (engine->context().get_constant(inputs[4])).toDouble();
    std::vector<nvinfer1::ITensor*> input_groups;
    std::vector<nvinfer1::ITensor*> output_groups;
    
    // divide input into num_group parts on channels
    // such as: 
    // input shape is [2, 10, 3, 3] and num_group = 2.
    // input is divided into 2 groups on channels, and each shape is [2, 5, 3, 3].
    std::vector<int64_t> start_vec(ori_shape_vec.size(), 0), size_vec(ori_shape_vec), stride_vec(ori_shape_vec.size(), 1);
    std::vector<int64_t> group_channel_rev_mask_vec(ori_shape_vec.size(), 1);
    std::vector<int64_t> group_channel_mask_vec(ori_shape_vec.size(), 0);
    group_channel_rev_mask_vec[1] = 0;
    group_channel_mask_vec[1] = 1;
    nvinfer1::Dims start_dims, size_dims, stride_dims;
    size_vec[1] = ori_shape_vec[1] / num_groups;
    if (is_dynamic) {
        for (size_t i = 0; i < size_vec.size(); i++) {
            size_vec[i] = 0;
        }
    }
    start_dims = sizes_to_nvdim(start_vec);
    size_dims = sizes_to_nvdim(size_vec);
    stride_dims = sizes_to_nvdim(stride_vec);
    nvinfer1::ITensor* ori_shape_tensor = nullptr;
    nvinfer1::ITensor* size_tensor = nullptr;
    nvinfer1::ITensor* start_tensor = nullptr;

    if (is_dynamic) {
        ori_shape_tensor = engine->network()->addShape(*input)->getOutput(0);
        at::Tensor group_channel_rev_mask_tensor = torch::tensor(group_channel_rev_mask_vec, torch::kInt);
        group_channel_rev_mask_tensor[1] = num_groups;
        size_tensor = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kDIV, 
                        ori_shape_tensor, 
                        tensor_to_const(engine, group_channel_rev_mask_tensor), 
                        (layer_info(node) + "_div_for_shape").c_str())->getOutput(0);
        at::Tensor group_channel_mask_tensor = torch::tensor(group_channel_mask_vec, torch::kInt);

        start_tensor = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kPROD, 
                        size_tensor, 
                        tensor_to_const(engine, group_channel_mask_tensor), 
                        (layer_info(node) + "_prod_for_shape").c_str())->getOutput(0);
    }

    for (int i = 0; i < num_groups; i++) {
        start_dims.d[1] = size_vec[1] * i;
        nvinfer1::ISliceLayer* slice_l = engine->network()->addSlice(*input, start_dims, size_dims, stride_dims);
        if (is_dynamic) {
            nvinfer1::ITensor* start_it_tensor = add_elementwise(engine, 
                                    nvinfer1::ElementWiseOperation::kPROD,
                                    start_tensor,
                                    tensor_to_const(engine, torch::tensor(i, torch::kInt)),
                                    (layer_info(node) + "_prod_for_start_" + std::to_string(i)).c_str())->getOutput(0);
            slice_l->setInput(1, *start_it_tensor);
            slice_l->setInput(2, *size_tensor);
            slice_l->setName((layer_info(node) + "_ISliceLayer_" + std::to_string(i)).c_str());
        }
        input_groups.push_back(slice_l->getOutput(0));
    }
    // calculate (x - E[x]) / sqrt((var + eps)) for each group
    for (size_t i = 0; i < input_groups.size(); i++) {
        // Set up axis_ask for E[x].
        uint32_t axis_mask = 0;
        for (size_t i = 0; i < ori_shape_vec.size() - 1; i++) {
            axis_mask |= 1 << (ori_shape_vec.size() - i - 1);
        }
        LOG(INFO) << "Axis Mask for E[x]" << std::bitset<32>(axis_mask);
        
        // E[x]
        nvinfer1::IReduceLayer* mean_expected = engine->network()->addReduce(*input_groups[i], 
                                                            nvinfer1::ReduceOperation::kAVG, axis_mask, true);
        POROS_CHECK(mean_expected, "Unable to create mean_expected from node: " << *node);
        mean_expected->setName((layer_info(node) + "_IReduceLayer(mean_expected)_" + std::to_string(i)).c_str());
        nvinfer1::ITensor* mean_expected_out = mean_expected->getOutput(0);

        // X-E[x]
        nvinfer1::ILayer* sub = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kSUB, 
                        input_groups[i], 
                        mean_expected_out, 
                        (layer_info(node) + "_sub_" + std::to_string(i)).c_str());
        POROS_CHECK(sub, "Unable to create Sub layer from node: " << *node);
        nvinfer1::ITensor* xsubmean_out = sub->getOutput(0);
        
        // Variance = mean(pow(xsubmean,2))
        float pow_scalar = 2.0;
        nvinfer1::ITensor* exponent = tensor_to_const(engine, torch::tensor({pow_scalar}, torch::kFloat));
        nvinfer1::ILayer* pow = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kPOW, 
                        xsubmean_out, 
                        exponent, 
                        (layer_info(node) + "_pow_" + std::to_string(i)).c_str());
        POROS_CHECK(pow, "Unable to create Pow layer from node: " << *node);
        nvinfer1::ITensor* pow_out = pow->getOutput(0);
        
        nvinfer1::IReduceLayer* mean_var = engine->network()->addReduce(*pow_out, 
                                                                nvinfer1::ReduceOperation::kAVG, axis_mask, true);
        POROS_CHECK(mean_var, "Unable to create mean_var from node: " << *node);
        mean_var->setName((layer_info(node) + "_IReduceLayer(mean_var)_" + std::to_string(i)).c_str());
        nvinfer1::ITensor* mean_var_out = mean_var->getOutput(0);
        
        // Variance + eps
        nvinfer1::ITensor* eps_tensor = tensor_to_const(engine, torch::tensor({eps}, torch::kFloat));
        nvinfer1::ILayer* add = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kSUM, 
                        mean_var_out, 
                        eps_tensor, 
                        (layer_info(node) + "_add_" + std::to_string(i)).c_str());
        POROS_CHECK(add, "Unable to create Add layer from node: " << *node);
        nvinfer1::ITensor* add_out = add->getOutput(0);
        
        // SQRT((Var + eps))
        nvinfer1::IUnaryLayer* sqrt = engine->network()->addUnary(*add_out, nvinfer1::UnaryOperation::kSQRT);
        POROS_CHECK(sqrt, "Unable to create unary(sqrt) from node: " << *node);
        sqrt->setName((layer_info(node) + "_IUnaryLayer_" + std::to_string(i)).c_str());
        nvinfer1::ITensor* sqrt_out = sqrt->getOutput(0);
        
        // (x - E[x]) / sqrt((var + eps))
        nvinfer1::ILayer* div = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kDIV, 
                        xsubmean_out, 
                        sqrt_out, 
                        (layer_info(node) + "_div_" + std::to_string(i)).c_str());
        POROS_CHECK(div, "Unable to create div layer from node: " << *node);
        nvinfer1::ITensor* div_out = div->getOutput(0);
        output_groups.push_back(div_out);
    }
    nvinfer1::IConcatenationLayer* cat_layer = engine->network()->addConcatenation(output_groups.data(), 
                                                                                    output_groups.size());
    cat_layer->setAxis(1);
    cat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
    nvinfer1::ITensor* cat_out = cat_layer->getOutput(0);
    engine->context().set_tensor(node->outputs()[0], cat_out);
    
    torch::jit::IValue maybe_weight = engine->context().get_constant(inputs[2]);
    torch::jit::IValue maybe_bias = engine->context().get_constant(inputs[3]);
    //when weight and bias setting is both None
    if (!maybe_weight.isTensor() && !maybe_bias.isTensor()) {
        engine->context().set_tensor(node->outputs()[0], cat_out);
        LOG(INFO) << "Output tensor shape: " << cat_out->getDimensions();
        return true;
    }
    
    /*------------------------------------------------------------
     * situation when weight or bias setting is not None
     * ------------------------------------------------------------*/
    // Remove batch dimension from input shape for expand_size, which will
    // be used to create weights for addScaleNd later.

    /** TODO: IS the first input size always are always be batch?????
    * if not, this converter is not ok。
    * */
    
    nvinfer1::ILayer* scale_l = nullptr;
    nvinfer1::ILayer* shift_l = nullptr;
    std::vector<int64_t> weights_dims_vec;
    std::vector<int64_t> weights_shuffle_dims_vec(ori_shape_vec.size() - 1, 1);
    weights_shuffle_dims_vec[0] = ori_shape_vec[1];
    weights_dims_vec.insert(weights_dims_vec.end(), ori_shape_vec.begin() + 1, ori_shape_vec.end());
    nvinfer1::ITensor* weights_shape_tensor = nullptr;
    // although shape of input is dynamic, its rank is fix. so we can remove its batch dim to get expand size.
    if (is_dynamic) {
        for (size_t i = 0; i < weights_dims_vec.size(); i++) {
            weights_dims_vec[i] = 0;
        }
        std::vector<int64_t> start = {1}, size = {ori_shape.nbDims - 1}, stride = {1};
        weights_shape_tensor = engine->network()->addSlice(*ori_shape_tensor, 
                                                            sizes_to_nvdim(start), 
                                                            sizes_to_nvdim(size), 
                                                            sizes_to_nvdim(stride))->getOutput(0);
    }
    // if gamma exist
    if (maybe_weight.isTensor()) {
        torch::Tensor gamma = maybe_weight.toTensor();
        nvinfer1::ITensor* gamma_tensor = tensor_to_const(engine, gamma);
        nvinfer1::ITensor* gamma_tensor_expand = expand_gamma_beta(engine, 
                                                                    gamma_tensor, 
                                                                    sizes_to_nvdim(weights_shuffle_dims_vec), 
                                                                    weights_shape_tensor, 
                                                                    sizes_to_nvdim(weights_dims_vec), 
                                                                    is_dynamic,
                                                                    layer_info(node) + "_ISliceLayer_for_gamma");
        scale_l = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kPROD, 
                        cat_out, 
                        gamma_tensor_expand, 
                        (layer_info(node) + "_prod_for_scale").c_str());
    }
    // if beta exist
    if (maybe_bias.isTensor()) {
        torch::Tensor ori_beta = maybe_bias.toTensor();
        nvinfer1::ITensor* beta_tensor = tensor_to_const(engine, ori_beta);
        nvinfer1::ITensor* beta_tensor_expand = expand_gamma_beta(engine, 
                                                                beta_tensor, 
                                                                sizes_to_nvdim(weights_shuffle_dims_vec), 
                                                                weights_shape_tensor, 
                                                                sizes_to_nvdim(weights_dims_vec), 
                                                                is_dynamic,
                                                                layer_info(node) + "_ISliceLayer_for_beta");
        if (scale_l == nullptr) {
            shift_l = add_elementwise(engine,
                        nvinfer1::ElementWiseOperation::kSUM,
                        cat_out,
                        beta_tensor_expand,
                        (layer_info(node) + "_sum_for_shift").c_str());

        } else {
            shift_l = add_elementwise(engine,
                        nvinfer1::ElementWiseOperation::kSUM,
                        scale_l->getOutput(0),
                        beta_tensor_expand,
                        (layer_info(node) + "_sum_for_shift").c_str());
        }
        nvinfer1::ITensor* shift_l_out = shift_l->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], shift_l_out);
        LOG(INFO) << "Output tensor shape: " << shift_l_out->getDimensions();
    } else {
        nvinfer1::ITensor* scale_l_out = scale_l->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], scale_l_out);
        LOG(INFO) << "Output tensor shape: " << scale_l_out->getDimensions();
        
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, GroupNormConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
