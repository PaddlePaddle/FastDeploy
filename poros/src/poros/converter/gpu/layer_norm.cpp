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
* @file layer_norm.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug 20 15:28:37 CST 2021
* @brief
**/

#include "poros/converter/gpu/layer_norm.h"
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
aten::layer_norm(Tensor input, 
int[] normalized_shape, 
Tensor? weight=None, 
Tensor? bias=None, 
float eps=1e-05, 
bool cudnn_enable=True) -> Tensor
*/
bool LayerNormConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 6), "invaid inputs size for LayerNormConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for LayerNormConverter is not Tensor as expected");
    // weight & bias
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for LayerNormConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[3]->node()->kind() == torch::jit::prim::Constant),
        "input[3] for LayerNormConverter is not come from prim::Constant as expected");

    nvinfer1::ITensor* input = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((input != nullptr), "Unable to init input tensor for node: " << *node);
    nvinfer1::Dims orig_shape = input->getDimensions();
    std::vector<int64_t> shape = nvdim_to_sizes(orig_shape);
    
    /* Layer_Norm normalizes over last N dimensions.
        normalizaed_shape could be (C,H,W), (H,W), or (W). */
    c10::List<int64_t> normalized_shape = (engine->context().get_constant(inputs[1])).toIntList();
    std::vector<int64_t> normalized_shape_vec = nvdim_to_sizes(sizes_to_nvdim(normalized_shape));
    
    // Unwrap eps.
    double eps = (engine->context().get_constant(inputs[4])).toDouble();

    // Set up  axis_ask for E[x].
    uint32_t axis_mask = 0;
    for (size_t i = 0; i < normalized_shape_vec.size(); i++) {
        axis_mask |= 1 << (shape.size() - i - 1);
    }
    LOG(INFO) << "Axis Mask for E[x]" << std::bitset<32>(axis_mask);
    
    // E[x]
    nvinfer1::IReduceLayer* mean_expected = engine->network()->addReduce(*input, 
                                                        nvinfer1::ReduceOperation::kAVG, axis_mask, true);
    POROS_CHECK(mean_expected, "Unable to create mean_expected from node: " << *node);
    mean_expected->setName((layer_info(node) + "_IReduceLayer(mean_expected)").c_str());
    nvinfer1::ITensor* mean_expected_out = mean_expected->getOutput(0);

    // X-E[x]
    nvinfer1::ILayer* sub = add_elementwise(engine, 
                    nvinfer1::ElementWiseOperation::kSUB, 
                    input, 
                    mean_expected_out, 
                    (layer_info(node) + "_sub").c_str());
    POROS_CHECK(sub, "Unable to create Sub layer from node: " << *node);
    nvinfer1::ITensor* xsubmean_out = sub->getOutput(0);
    
    // Variance = mean(pow(xsubmean,2))
    float pow_scalar = 2;
    nvinfer1::ITensor* exponent = tensor_to_const(engine, torch::tensor({pow_scalar}));
    nvinfer1::ILayer* pow = add_elementwise(engine, 
                    nvinfer1::ElementWiseOperation::kPOW, 
                    xsubmean_out, 
                    exponent, 
                    (layer_info(node) + "_pow").c_str());
    POROS_CHECK(pow, "Unable to create Pow layer from node: " << *node);
    nvinfer1::ITensor* pow_out = pow->getOutput(0);
    
    nvinfer1::IReduceLayer* mean_var = engine->network()->addReduce(*pow_out, 
                                                            nvinfer1::ReduceOperation::kAVG, axis_mask, true);
    POROS_CHECK(mean_var, "Unable to create mean_var from node: " << *node);
    mean_var->setName((layer_info(node) + "_IReduceLayer(mean_var)").c_str());
    nvinfer1::ITensor* mean_var_out = mean_var->getOutput(0);
    
    // Variance + eps
    nvinfer1::ITensor* eps_tensor = tensor_to_const(engine, torch::tensor({eps}));
    nvinfer1::ILayer* add = add_elementwise(engine, 
                    nvinfer1::ElementWiseOperation::kSUM, 
                    mean_var_out, 
                    eps_tensor, 
                    (layer_info(node) + "_sum").c_str());
    POROS_CHECK(add, "Unable to create Add layer from node: " << *node);
    nvinfer1::ITensor* add_out = add->getOutput(0);
    
    // SQRT((Var + eps))
    nvinfer1::IUnaryLayer* sqrt = engine->network()->addUnary(*add_out, nvinfer1::UnaryOperation::kSQRT);
    POROS_CHECK(sqrt, "Unable to create unary(sqrt) from node: " << *node);
    sqrt->setName((layer_info(node) + "_IUnaryLayer").c_str());
    nvinfer1::ITensor* sqrt_out = sqrt->getOutput(0);
    
    // (x - E[x]) / sqrt((var + eps))
    nvinfer1::ILayer* div = add_elementwise(engine, 
                    nvinfer1::ElementWiseOperation::kDIV, 
                    xsubmean_out, 
                    sqrt_out, 
                    (layer_info(node) + "_div").c_str());
    POROS_CHECK(div, "Unable to create div layer from node: " << *node);
    nvinfer1::ITensor* div_out = div->getOutput(0);
    
    torch::jit::IValue maybe_weight = engine->context().get_constant(inputs[2]);
    torch::jit::IValue maybe_bias = engine->context().get_constant(inputs[3]);
    //when weight and bias setting is both None
    if (!maybe_weight.isTensor() && !maybe_bias.isTensor()) {
        engine->context().set_tensor(node->outputs()[0], div_out);
        LOG(INFO) << "Output tensor shape: " << div_out->getDimensions();
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
    
    // Set up gamma and beta by tensor_to_const directly, 
    // boardcast will be done automatically when add_elementwise, so need not expand
    nvinfer1::ILayer* scale_l = nullptr;
    nvinfer1::ILayer* shift_l = nullptr;
    if (maybe_weight.isTensor()) {
        torch::Tensor gamma = maybe_weight.toTensor();
        nvinfer1::ITensor* gamma_tensor = tensor_to_const(engine, gamma);
        scale_l = add_elementwise(engine, 
                        nvinfer1::ElementWiseOperation::kPROD, 
                        div_out, 
                        gamma_tensor, 
                        (layer_info(node) + "_prod_for_gamma").c_str());
    }
    
    if (maybe_bias.isTensor()) {
        torch::Tensor ori_beta = maybe_bias.toTensor();
        nvinfer1::ITensor* beta_tensor = tensor_to_const(engine, ori_beta);
        if (scale_l == nullptr) {
            shift_l = add_elementwise(engine,
                        nvinfer1::ElementWiseOperation::kSUM,
                        div_out,
                        beta_tensor,
                        (layer_info(node) + "_sum_for_beta").c_str());

        } else {
            shift_l = add_elementwise(engine,
                        nvinfer1::ElementWiseOperation::kSUM,
                        scale_l->getOutput(0),
                        beta_tensor,
                        (layer_info(node) + "_sum_for_beta").c_str());
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

POROS_REGISTER_CONVERTER(TensorrtEngine, LayerNormConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
