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
* @file reduce.cpp
* @author tianjinjin@baidu.com
* @date Fri Aug 27 10:18:24 CST 2021
* @brief 
**/

#include "poros/converter/gpu/reduce.h"
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
"aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
"aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"*/
bool MeanConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for MeanConverter is not Tensor as expected");


    //extract self
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_dims = nvdim_to_sizes(in_tensor->getDimensions());
    LOG(WARNING) << "MeanConverter disregards dtype";

    uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);
    auto keepdim = false;

    // aten::mean.dim situation
    auto maybe_dim = engine->context().get_constant(inputs[1]);
    if ((inputs.size() == 4) &&  maybe_dim.isIntList() && 
          (engine->context().get_constant(inputs[2])).isBool()) {
        auto dims = maybe_dim.toIntList();
        c10::List<int64_t> calculated_dims;
        for (size_t i = 0; i < dims.size(); i++) {
            auto dim_val = dims[i] < 0 ? (in_dims.size() + dims[i]) : dims[i];
            calculated_dims.push_back(dim_val);
        }
        axis_mask = 0;
        for (size_t d = 0; d < calculated_dims.size(); d++) {
            axis_mask |= 1 << calculated_dims[d];
        }
        keepdim = (engine->context().get_constant(inputs[2])).toBool();
    }

    auto mean_layer = engine->network()->addReduce(*in_tensor, 
                    nvinfer1::ReduceOperation::kAVG, axis_mask, keepdim);
    POROS_CHECK(mean_layer, "Unable to create mean layer from node: " << *node);
    mean_layer->setName((layer_info(node) + "_IReduceLayer_avg").c_str());

    engine->context().set_tensor(node->outputs()[0], mean_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << mean_layer->getOutput(0)->getDimensions();
    return true;
}


/*
"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
"aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"*/
bool SumConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SumConverter is not Tensor as expected");

    //extract self
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_dims = nvdim_to_sizes(in_tensor->getDimensions());
    LOG(WARNING) << "SumConverter disregards dtype";

    uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);
    auto keepdim = false;

    // aten::sum.dim_IntList situation
    auto maybe_dim = engine->context().get_constant(inputs[1]);
    if ((inputs.size() == 4) &&  maybe_dim.isIntList() && 
          (engine->context().get_constant(inputs[2])).isBool()) {
        auto dims = maybe_dim.toIntList();
        c10::List<int64_t> calculated_dims;
        for (size_t i = 0; i < dims.size(); i++) {
            auto dim_val = dims[i] < 0 ? (in_dims.size() + dims[i]) : dims[i];
            calculated_dims.push_back(dim_val);
        }
        axis_mask = 0;
        for (size_t d = 0; d < calculated_dims.size(); d++) {
            axis_mask |= 1 << calculated_dims[d];
        }
        keepdim = (engine->context().get_constant(inputs[2])).toBool();
    }

    auto mean_layer = engine->network()->addReduce(*in_tensor, 
                    nvinfer1::ReduceOperation::kSUM, axis_mask, keepdim);
    POROS_CHECK(mean_layer, "Unable to create mean layer from node: " << *node);
    mean_layer->setName((layer_info(node) + "_IReduceLayer_sum").c_str());

    engine->context().set_tensor(node->outputs()[0], mean_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << mean_layer->getOutput(0)->getDimensions();
    return true;
}

/*
"aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
"aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"*/
bool ProdConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ProdConverter is not Tensor as expected");

    //extract self
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_dims = nvdim_to_sizes(in_tensor->getDimensions());
    LOG(WARNING) << "ProdConverter disregards dtype";

    uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);
    auto keepdim = false;

    //aten::prod.dim_int situation
    auto maybe_dim = engine->context().get_constant(inputs[1]);
    if ((inputs.size() == 4) &&  maybe_dim.isInt() && 
          (engine->context().get_constant(inputs[2])).isBool()) {
        auto dim = maybe_dim.toInt();
        dim = dim < 0 ? (in_tensor->getDimensions().nbDims + dim) : dim;
        axis_mask = 1 << dim;

        keepdim = (engine->context().get_constant(inputs[2])).toBool();
    }

    auto mean_layer = engine->network()->addReduce(*in_tensor, 
                    nvinfer1::ReduceOperation::kPROD, axis_mask, keepdim);
    POROS_CHECK(mean_layer, "Unable to create mean layer from node: " << *node);
    mean_layer->setName((layer_info(node) + "_IReduceLayer_prod").c_str());

    engine->context().set_tensor(node->outputs()[0], mean_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << mean_layer->getOutput(0)->getDimensions();
    return true;
}

/*
"aten::max(Tensor self) -> Tensor",
"aten::max.other(Tensor self, Tensor other) -> Tensor"
"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"*/
bool MaxMinConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for MaxMinConverter is not Tensor as expected");

    //extract self
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_dims = nvdim_to_sizes(in_tensor->getDimensions());

    bool is_dynamic = check_nvtensor_is_dynamic(in_tensor);

    nvinfer1::ILayer* new_layer;
    //aten::max situation
    if (inputs.size() == 1) {
        uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);
        auto keepdim = false;

        nvinfer1::ReduceOperation reduce_type = (node->kind() == torch::jit::aten::max)
                                            ? nvinfer1::ReduceOperation::kMAX
                                            : nvinfer1::ReduceOperation::kMIN;
        new_layer = engine->network()->addReduce(*in_tensor, reduce_type, axis_mask, keepdim);
        new_layer->setName((layer_info(node) + "_IReduceLayer_max_or_min").c_str());
        POROS_CHECK(new_layer, "Unable to create reduce layer from node: " << *node);

    //aten::max.other situation
    } else if (inputs.size() == 2) {
        //extract other
        auto other = engine->context().get_tensor(inputs[1]);
        POROS_CHECK_TRUE((other != nullptr), "Unable to init input tensor for node: " << *node);

        nvinfer1::ElementWiseOperation element_type = (node->kind() == torch::jit::aten::max)
                                            ? nvinfer1::ElementWiseOperation::kMAX
                                            : nvinfer1::ElementWiseOperation::kMIN;
        new_layer = add_elementwise(engine,
                            element_type,
                            in_tensor,
                            other,
                            layer_info(node) + "_max_or_min");
        POROS_CHECK(new_layer, "Unable to create element_wise layer from node: " << *node);

    } else if (inputs.size() == 3 && node->outputs().size() == 2 &&
                inputs[1]->type()->kind() == c10::TypeKind::IntType) {
        POROS_CHECK_TRUE((in_dims.size() > 1), 
            "Converter aten::max.dim error: At least 2 dimensions are required for input[0].");
        nvinfer1::ITensor* output_max = nullptr;
        nvinfer1::ITensor* output_indices = nullptr;
        int64_t dim = engine->context().get_constant(inputs[1]).toInt();
        dim = dim < 0 ? in_dims.size() + dim : dim;

        bool keep_dim = engine->context().get_constant(inputs[2]).toBool();
        uint32_t shiftDim = 1 << dim;
        nvinfer1::TopKOperation topk_option = (node->kind() == torch::jit::aten::max) ?
                                                nvinfer1::TopKOperation::kMAX : 
                                                nvinfer1::TopKOperation::kMIN;
        nvinfer1::ITopKLayer* topk_layer =  engine->network()->addTopK(*in_tensor, topk_option, 1, shiftDim);
        POROS_CHECK(topk_layer, "Unable to create TopK layer from node: " << *node);
        topk_layer->setName((layer_info(node) + "_ITopKLayer").c_str());
        output_max = topk_layer->getOutput(0);
        output_indices = topk_layer->getOutput(1);
        
        // squeeze output dim
        if (in_tensor->getDimensions().nbDims > 1 && !keep_dim) {
            auto shuffle_layer1 = engine->network()->addShuffle(*output_max);
            auto shuffle_layer2 = engine->network()->addShuffle(*output_indices);
            if (is_dynamic) {
                nvinfer1::ITensor* self_shape_tensor = engine->network()->addShape(*in_tensor)->getOutput(0);
                nvinfer1::ITensor* squeeze_output_shape = squeeze_nv_shapetensor(engine, self_shape_tensor, dim);
                shuffle_layer1->setInput(1, *squeeze_output_shape);
                shuffle_layer2->setInput(1, *squeeze_output_shape);
            } else {
                in_dims.erase(in_dims.begin() + dim);
                nvinfer1::Dims squeeze_output_dims = sizes_to_nvdim(in_dims);
                shuffle_layer1->setReshapeDimensions(squeeze_output_dims);
                shuffle_layer2->setReshapeDimensions(squeeze_output_dims);
            }
            output_max = shuffle_layer1->getOutput(0);
            output_indices = shuffle_layer2->getOutput(0);
        }

        engine->context().set_tensor(node->outputs()[0], output_max);
        engine->context().set_tensor(node->outputs()[1], output_indices);
        LOG(INFO) << "Output tensor1 shape: " << output_max->getDimensions();
        LOG(INFO) << "Output tensor2 shape: " << output_indices->getDimensions();
        return true;

    } else{
        //some other situation not supported yet
        POROS_THROW_ERROR("We should never reach here for MaxMinConverter, meet Unsupported inputs size!");
    }
    
    nvinfer1::ITensor* output = new_layer->getOutput(0);

    if (output->getDimensions().nbDims == 0) {
        auto shuffle_layer = engine->network()->addShuffle(*output);
        nvinfer1::Dims output_dims;
        output_dims.nbDims = 1;
        output_dims.d[0] = 1;
        shuffle_layer->setReshapeDimensions(output_dims);
        shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_output").c_str());
        output = shuffle_layer->getOutput(0);
    }
    engine->context().set_tensor(node->outputs()[0], output);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

/*
"aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)"*/
bool ArgmaxArgminConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ArgmaxArgminConverter is not Tensor as expected");

    // TODO: to imp dim=None
    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::IntType::get())), 
        "input[1] for ArgmaxArgminConverter is not int as expected");

    //extract self
    auto in_tensor = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in_tensor != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_dims = nvdim_to_sizes(in_tensor->getDimensions());

    bool is_dynamic = check_nvtensor_is_dynamic(in_tensor);

    POROS_CHECK_TRUE((in_dims.size() > 1), 
        "Converter aten::argmax error: At least 2 dimensions are required for input[0].");
    nvinfer1::ITensor* output_indices = nullptr;

    int64_t dim = 0;
    dim = engine->context().get_constant(inputs[1]).toInt();
    dim = dim < 0 ? in_dims.size() + dim : dim;
    bool keep_dim = engine->context().get_constant(inputs[2]).toBool();
    uint32_t shiftDim = 1 << dim;

    // nvinfer1::TopKOperation noly support kFLOAT, so this is transfer kINT32 to kFLOAT
    if (in_tensor->getType() == nvinfer1::DataType::kINT32) {
        auto id_layer = engine->network()->addIdentity(*in_tensor);
        id_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        id_layer->setName((layer_info(node) + "_IIdentityLayer_int32_to_float").c_str());
        in_tensor = id_layer->getOutput(0);
    }

    nvinfer1::TopKOperation topk_option = (node->kind() == torch::jit::aten::argmax) ?
                                            nvinfer1::TopKOperation::kMAX : 
                                            nvinfer1::TopKOperation::kMIN;
    nvinfer1::ITopKLayer* topk_layer =  engine->network()->addTopK(*in_tensor, topk_option, 1, shiftDim);
    POROS_CHECK(topk_layer, "Unable to create TopK layer from node: " << *node);
    topk_layer->setName((layer_info(node) + "_ITopKLayer").c_str());
    output_indices = topk_layer->getOutput(1);

    // squeeze output dim
    if (in_tensor->getDimensions().nbDims > 1 && !keep_dim) {
        auto shuffle_layer = engine->network()->addShuffle(*output_indices);
        if (is_dynamic) {
            nvinfer1::ITensor* self_shape_tensor = engine->network()->addShape(*in_tensor)->getOutput(0);
            nvinfer1::ITensor* squeeze_output_shape = squeeze_nv_shapetensor(engine, self_shape_tensor, dim);
            shuffle_layer->setInput(1, *squeeze_output_shape);
        } else {
            in_dims.erase(in_dims.begin() + dim);
            nvinfer1::Dims squeeze_output_dims = sizes_to_nvdim(in_dims);
            shuffle_layer->setReshapeDimensions(squeeze_output_dims);
        }
        output_indices = shuffle_layer->getOutput(0);
    }
    engine->context().set_tensor(node->outputs()[0], output_indices);
    LOG(INFO) << "Output tensor shape: " << output_indices->getDimensions();
    return true;
}


POROS_REGISTER_CONVERTER(TensorrtEngine, MeanConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, SumConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ProdConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, MaxMinConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ArgmaxArgminConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
