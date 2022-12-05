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
* @file linear.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug 20 17:21:44 CST 2021
* @brief 
**/

#include "poros/converter/gpu/linear.h"
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

/** aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
 * the implementation of aten::linear in pytorch is in file: aten/src/Aten/native/Linear.cpp
 * the core function is like this:
 *  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);
    if (input.dim() == 2 && bias->defined()) {
        return at::addmm(*bias, input, weight.t());
    }
    auto output = at::matmul(input, weight.t());
    if (bias->defined()) {
        output.add_(*bias);
    }
    return output;
* we can refer to the implement of original pytorch.
* ******************************
* %res = aten::linear(%input, %weight_0, %bias)
* try to converter matmul like below:
*
* %weight = aten::t(%weight_0)
* %mm  = aten::matmul(%input, %weight)
* if %bias is None:
*    return %mm
* else:
*   if (input.dim == 2):
*        %res = aten::add(%bias, %mm, 1)
*   else:
*        %res = aten::add(%mm, %bias, 1)
**/
bool LinearConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for LinearConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for LinearConverter is not Tensor as expected");

    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    //auto origin_self_dim = self->getDimensions().nbDims;
    
    // handle weight
    nvinfer1::ITensor* weight = nullptr;
    bool need_trans = false;
    auto maybe_weight = engine->context().get_constant(inputs[1]);
    if (maybe_weight.isTensor()) {
        //常量tensor
        at::Tensor weight_t = maybe_weight.toTensor().t();
        int weight_rank = weight_t.sizes().size();
        //需要padding tensor 的情况，直接转置并padding完成后，再转constant_tensor, 避免命中tensorrt中constshuffle的tatic.
        if (weight_rank < self->getDimensions().nbDims) {
            at::Tensor padding_weight  = weight_t;
            for (int dim = weight_rank; dim < self->getDimensions().nbDims; ++dim) {
                padding_weight = weight_t.unsqueeze(0);
            }
            weight = tensor_to_const(engine, padding_weight);
        } else {
            weight = tensor_to_const(engine, weight_t);
        }
    } else {
        //weight 来自其他的tensor
        weight = engine->context().get_tensor(inputs[1]);
        if (weight->getDimensions().nbDims >= 2) {
            need_trans = true;
        }
        /*  //转置交给matmul, 不再自己shuffle实现。
        auto weight_before_trans = engine->context().get_tensor(inputs[1]);
        auto weight_dims = weight_before_trans->getDimensions();
        if (weight_dims.nbDims < 2) {
            weight = weight_before_trans;
        } else {
            //like aten::transpose(input, 0, 1)
            auto shuffle_layer = engine->network()->addShuffle(*weight_before_trans);
            POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
            nvinfer1::Permutation first_perm;
            first_perm.order[0] = 1;
            first_perm.order[1] = 0;
            shuffle_layer->setFirstTranspose(first_perm);
            shuffle_layer->setZeroIsPlaceholder(false);
            shuffle_layer->setName((layer_info(node) + "_IShuffleLayer(weight_transpose)").c_str());
            weight = shuffle_layer->getOutput(0);
        } */
    }
    
    // Ensure self and weight tensors have same nbDims by expanding the dimensions (from 0 axis) if
    // necessary.
    if (self->getDimensions().nbDims < weight->getDimensions().nbDims) {
        self = add_padding(engine, node, self, weight->getDimensions().nbDims, false, false);
    } else {
        weight = add_padding(engine, node, weight, self->getDimensions().nbDims, false, false);
    }
    
    nvinfer1::IMatrixMultiplyLayer* mm_layer = nullptr;
    if (need_trans == true) {
        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kNONE, *weight, nvinfer1::MatrixOperation::kTRANSPOSE);
    } else {
        mm_layer = engine->network()->addMatrixMultiply(
            *self, nvinfer1::MatrixOperation::kNONE, *weight, nvinfer1::MatrixOperation::kNONE);
    }
    POROS_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *node);
    
    auto bias = engine->context().get_tensor(inputs[2]);
    /*--------------------------------------------------------------
     *               bias is None situation
     * -------------------------------------------------------------*/
    //bias is None situation return directly
    if (bias == nullptr) {
        mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer").c_str());
        engine->context().set_tensor(node->outputs()[0], mm_layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << mm_layer->getOutput(0)->getDimensions();
        return true;
    }

    /*--------------------------------------------------------------
     *               bias is not None situation
     * -------------------------------------------------------------*/
    mm_layer->setName((layer_info(node) + "_IMatrixMultiplyLayer").c_str());

    nvinfer1::ILayer* new_layer = nullptr;
    // if (origin_self_dim == 2) {
    //     //TODO: ADD SOME FUNCTION HERE
    // } else {
    new_layer = add_elementwise(engine, 
        nvinfer1::ElementWiseOperation::kSUM,
        mm_layer->getOutput(0),
        bias,
        layer_info(node) + "_sum");
    //}
    POROS_CHECK(new_layer, "Unable to create add layer from node: " << *node);
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));

    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
}

//DEPRECATED: result do not match the pytorch output
bool LinearConverter::converter_fully_connect_version(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for LinearConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for LinearConverter is not Tensor as expected");
    // weight & bias
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::TensorType::get())),
        "input[1] for LinearConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for LinearConverter is not come from prim::Constant as expected");

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto shape = nvdim_to_sizes(in->getDimensions());
    LOG(INFO) << "Input tensor shape: " << in->getDimensions();

    // PyTorch follows in: Nx*xIN, W: OUTxIN, B: OUT, out: Nx*xOUT
    // TensorRT inserts a flatten in when following conv
    POROS_ASSERT(shape.size() >= 2,
        "aten::linear expects input tensors to be of shape [N,..., in features], but found input Tensor less than 2D");
        
    if (shape.size() < 4) {
        // Flatten
        std::vector<int64_t> new_shape;
        new_shape.push_back(shape[0]);
        new_shape.push_back(1);
        new_shape.push_back(1);
        new_shape.push_back(nvdim_to_volume(sizes_to_nvdim(shape)) / shape[0]);
        auto new_dims = sizes_to_nvdim(new_shape);
        
        LOG(INFO) << "Input shape is less than 4D got: " << sizes_to_nvdim(shape)
                << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_dims;
        
        auto in_shuffle = engine->network()->addShuffle(*in);
        in_shuffle->setReshapeDimensions(new_dims);
        in_shuffle->setName((layer_info(node) + "_IShuffleLayer").c_str());
        in = in_shuffle->getOutput(0);
    }

    auto w_tensor = (engine->context().get_constant(inputs[1])).toTensor();
    Weights w = Weights(w_tensor);
    
    nvinfer1::ILayer* new_layer;
    auto maybe_bias = engine->context().get_constant(inputs[2]);
    if (maybe_bias.isTensor()) {
        auto bias = maybe_bias.toTensor();
        Weights b = Weights(bias);
        new_layer = engine->network()->addFullyConnected(*in, w.outputs_num, w.data, b.data);
    } else {
        LOG(INFO) << "There is no bias for the linear layer";
        new_layer = engine->network()->addFullyConnected(*in, w.outputs_num, w.data, Weights().data);
    }
    POROS_CHECK(new_layer, "Unable to create linear layer from node: " << *node);
    new_layer->setName((layer_info(node) + "_IFullyConnectedLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], new_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << new_layer->getOutput(0)->getDimensions();
    return true;
} 

POROS_REGISTER_CONVERTER(TensorrtEngine, LinearConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
