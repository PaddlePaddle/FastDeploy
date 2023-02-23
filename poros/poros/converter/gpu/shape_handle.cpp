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
* @file shape_handle.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/shape_handle.h"
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
"aten::size(Tensor self) -> (int[])
aten::size.int(Tensor self, int dim) -> int
"*/
bool AtenSizeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1 || inputs.size() == 2), "invaid inputs size for AtenSizeConverter");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    auto shape = engine->network()->addShape(*self);
    POROS_CHECK(shape, "Unable to create shape layer from node: " << *node);
    shape->setName((layer_info(node) + "_IShapeLayer_for_self").c_str());
    auto shape_out = shape->getOutput(0);
    
    //output is int[] situation
    if (inputs.size() == 1) {
        LOG(INFO) << "start converter aten::size(Tensor self) -> (int[])";
        engine->context().set_tensor(node->outputs()[0], shape_out);
        LOG(INFO) << "Output tensor shape: " << shape_out->getDimensions();
    //output is int situation
    } else {
        LOG(INFO) << "start converter aten::size.int(Tensor self, int dim) -> int";
        auto dim = (engine->context().get_constant(inputs[1])).toInt();
        nvinfer1::Dims self_dims = self->getDimensions();
        dim = dim < 0 ? dim + self_dims.nbDims : dim;

        //extract the specific dynamic dim as a 1D-1value tensor
        std::vector<int64_t> start_vec{dim}, size_vec{1}, stride_vec{1};
        auto size_layer = engine->network()->addSlice(*shape_out,
                                                sizes_to_nvdim(start_vec),
                                                sizes_to_nvdim(size_vec),
                                                sizes_to_nvdim(stride_vec));
        POROS_CHECK(size_layer, "Unable to given dim info from node: " << *node);
        auto size_out = size_layer->getOutput(0);
        size_layer->setName((layer_info(node) + "_ISliceLayer_for_size").c_str());
        engine->context().set_tensor(node->outputs()[0], size_out);
        LOG(INFO) << "Output tensor shape: " << size_out->getDimensions();
    }
    return true;
}

bool ShapeastensorConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for ShapeastensorConverter");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    auto shape = engine->network()->addShape(*self);
    POROS_CHECK(shape, "Unable to create shape layer from node: " << *node);
    shape->setName((layer_info(node) + "_IShapeLayer_for_self").c_str());
    auto shape_out = shape->getOutput(0);
    
    engine->context().set_tensor(node->outputs()[0], shape_out);
    LOG(INFO) << "Output tensor shape: " << shape_out->getDimensions();

    return true;
}

// aten::len.Tensor(Tensor t) -> (int)
// aten::len.t(t[] a) -> (int)
bool LenConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for LenConverter");

    // extract self
    auto self = engine->context().get_tensor(inputs[0]);
    // POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    if (self != nullptr) {
        nvinfer1::Dims self_dims = self->getDimensions();
        if (self_dims.nbDims == 0) {
            engine->context().set_constant(node->outputs()[0], 0);
        } else if (self_dims.nbDims > 0 && self_dims.d[0] >= 0) {
            engine->context().set_constant(node->outputs()[0], self_dims.d[0]);
        } else {
            // dynamic
            nvinfer1::ITensor* self_shape = engine->network()->addShape(*self)->getOutput(0);
            self_shape->setName((layer_info(node) + "_IShapeLayer_for_self").c_str());

            std::vector<int64_t> start_vec{0}, size_vec{1}, stride_vec{1};
            auto slice_layer = engine->network()->addSlice(*self_shape,
                                                    sizes_to_nvdim(start_vec),
                                                    sizes_to_nvdim(size_vec),
                                                    sizes_to_nvdim(stride_vec));
            POROS_CHECK(slice_layer, "Unable to given dim info from node: " << *node);
            slice_layer->setName((layer_info(node) + "_ISliceLayer_for_len").c_str());
            auto len_tensor = slice_layer->getOutput(0);
            engine->context().set_tensor(node->outputs()[0], len_tensor);
            LOG(INFO) << "Output tensor shape: " << len_tensor->getDimensions();
        }
    } else {
        // tensorlist
        if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors())) {
            std::vector<nvinfer1::ITensor*> output_vec;
            if (engine->context().get_tensorlist(inputs[0], output_vec)) {
                engine->context().set_constant(node->outputs()[0], int(output_vec.size()));
            } else {
                auto in_const = engine->context().get_constant(inputs[0]);
                engine->context().set_constant(node->outputs()[0], int(in_const.toList().size()));
            }
        // scalarlist
        } else if (inputs[0]->type()->isSubtypeOf(c10::ListType::ofInts()) ||
                inputs[0]->type()->isSubtypeOf(c10::ListType::ofFloats()) ||
                inputs[0]->type()->isSubtypeOf(c10::ListType::ofBools())) {
            auto in_const = engine->context().get_constant(inputs[0]);
            engine->context().set_constant(node->outputs()[0], int(in_const.toList().size()));
        } else {
            POROS_THROW_ERROR("Meet some unsupported output value type in LenConverter" << *node);
        }
    }
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, AtenSizeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ShapeastensorConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, LenConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
