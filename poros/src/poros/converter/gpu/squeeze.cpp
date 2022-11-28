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
* @file squeeze.cpp
* @author tianjinjin@baidu.com
* @date Wed Sep  1 11:19:13 CST 2021
* @brief 
**/

#include "poros/converter/gpu/squeeze.h"
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

nvinfer1::IShuffleLayer* add_shuffle_layer(TensorrtEngine* engine, const torch::jit::Node *node, \
                                                nvinfer1::ITensor* input, int64_t dim, int64_t idx) {
    auto shuffle_layer = engine->network()->addShuffle(*input);
    POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
    shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_index_" + std::to_string(idx)).c_str());
    nvinfer1::ITensor* input_shape_tensor = (engine->network()->addShape(*input))->getOutput(0);
    nvinfer1::ITensor* reshape_tensor = squeeze_nv_shapetensor(engine, input_shape_tensor, dim);
    
    if (reshape_tensor != nullptr) {
        shuffle_layer->setInput(1, *reshape_tensor);
    } else {
        LOG(INFO) << "squeeze nv shape tensor error!";
        return nullptr;
    }
    return shuffle_layer;
}


/*
"aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
https://pytorch.org/docs/stable/generated/torch.squeeze.html
将输入张量形状中的1去除并返回。 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)*/
bool SqueezeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2 || inputs.size() == 1), "invaid inputs size for SqueezeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SqueezeConverter is not Tensor as expected");
    if (inputs.size() == 2) {
        POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
            "input[1] for SqueezeConverter is not come from prim::Constant as expected");
    }

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    std::vector<int64_t> dims;
    int64_t sign = 0;
    if (inputs.size() == 1) {
        // 这里目前只支持非dynamic
        auto shape = self->getDimensions().d;
        for (int i = 0; i < self->getDimensions().nbDims; i++) {
            if (shape[i] == 1) {
                dims.push_back(i - sign);
                sign += 1;
            }
        }
        if (dims.size() == 0) {
            return true;
        }
    }
    else {
        //extract dim
        auto dim = (engine->context().get_constant(inputs[1])).toInt();
        auto self_dim = nvdim_to_sizes(self->getDimensions());
        if (dim < 0) {
            dim = self_dim.size() + dim;
        }

        if (self_dim[dim] != 1) {
            //不需要squeeze的情况
            engine->context().set_tensor(node->outputs()[0], self);
            LOG(INFO) << "Output tensor shape: " << self->getDimensions();
            return true;
        } else {
            dims = {dim};
        }
    }
    
    bool is_dynamic = check_nvtensor_is_dynamic(self);
    nvinfer1::IShuffleLayer* shuffle_layer = nullptr;
    if (is_dynamic) {
        shuffle_layer = add_shuffle_layer(engine, node, self, dims[0], 0);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node)
        if (nullptr == shuffle_layer){
            LOG(INFO) << "unsqueeze nv shape tensor error!";
            return false;
        } 
        for (size_t i = 1; i < dims.size(); i++) {
            shuffle_layer = add_shuffle_layer(engine, node, shuffle_layer->getOutput(0), dims[i], i);
            POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node)
        }
        engine->context().set_tensor(node->outputs()[0], shuffle_layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();
    } else {
        shuffle_layer = engine->network()->addShuffle(*self);
        POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
        shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
        for (size_t i = 0; i < dims.size(); i++) {
            if (i == 0) {
                shuffle_layer->setReshapeDimensions(squeeze_dims(self->getDimensions(), dims[i]));
            } else {
                shuffle_layer->setReshapeDimensions(squeeze_dims(shuffle_layer->getOutput(0)->getDimensions(), dims[i]));
            }
            
            if (i != dims.size() - 1) {
                shuffle_layer = engine->network()->addShuffle(*shuffle_layer->getOutput(0));
                shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_output").c_str());
            }
        }
    }
    engine->context().set_tensor(node->outputs()[0], shuffle_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();
    return true;
}

/*
"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
https://pytorch.org/docs/stable/generated/torch.unsqueeze.html*/
bool UnSqueezeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for UnSqueezeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for UnSqueezeConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for UnSqueezeConverter is not come from prim::Constant as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    //extract dim
    auto dim = (engine->context().get_constant(inputs[1])).toInt();
    if (self->getDimensions().nbDims == 0 && dim == 0) {
        auto shuffle_layer = engine->network()->addShuffle(*self);
        nvinfer1::Dims unsqueeze_dim;
        unsqueeze_dim.nbDims = 1;
        unsqueeze_dim.d[0] = 1; 
        shuffle_layer->setReshapeDimensions(unsqueeze_dim);
        shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
        auto output = shuffle_layer->getOutput(0);
        engine->context().set_tensor(node->outputs()[0], output);
        LOG(INFO) << "Output tensor shape: " << output->getDimensions();
        return true;
    }
    auto self_dim = nvdim_to_sizes(self->getDimensions());
    int64_t nbDims = self_dim.size();
    POROS_CHECK((dim <= nbDims && dim >= -(nbDims + 1)), 
        "Dimension out of range (expected to be in range of [" << -(nbDims + 1)
        << ", " << nbDims << "], but got " << dim << ")");
    if (dim < 0) {
        dim = self_dim.size() + dim + 1;
    }
        
    auto shuffle_layer = engine->network()->addShuffle(*self);
    POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
    bool is_dynamic = check_nvtensor_is_dynamic(self);
    if (is_dynamic) {
        nvinfer1::ITensor* input_shape_tensor = (engine->network()->addShape(*self))->getOutput(0);
        nvinfer1::ITensor* reshape_tensor = unsqueeze_nv_shapetensor(engine, input_shape_tensor, dim);
        if (reshape_tensor != nullptr) {
            shuffle_layer->setInput(1, *reshape_tensor);
        } else {
            LOG(INFO) << "unsqueeze nv shape tensor error!";
            return false;
        }
    } else {
        shuffle_layer->setReshapeDimensions(unsqueeze_dims(self->getDimensions(), dim));
    }
    shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_self").c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, SqueezeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, UnSqueezeConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
