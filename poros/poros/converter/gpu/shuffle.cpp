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

// Part of the following code in this file refs to
// https://github.com/pytorch/TensorRT/blob/master/core/conversion/converters/impl/shuffle.cpp
//
// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the 3-Clause BSD License

/**
* @file shuffle.cpp
* @author tianjinjin@baidu.com
* @date Wed Aug 18 16:23:29 CST 2021
* @brief 
**/

#include "poros/converter/gpu/converter_util.h"
#include "poros/converter/gpu/shuffle.h"
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
 * aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
 * **/
bool FlattenConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for FlattenConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for FlattenConverter is not Tensor as expected");
    //assumes int inputs are all come from prim::Constant.
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for FlattenConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for FlattenConverter is not come from prim::Constant as expected");
        
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);

    auto start_dim = (engine->context().get_constant(inputs[1])).toInt();
    auto end_dim = (engine->context().get_constant(inputs[2])).toInt();

    auto in_shape = nvdim_to_sizes(in->getDimensions());
    auto in_shape_rank = in_shape.size();
    // 倒序转正序
    start_dim = start_dim < 0 ? start_dim + in_shape_rank : start_dim;
    end_dim = end_dim < 0 ? end_dim + in_shape_rank : end_dim;

    POROS_CHECK_TRUE((start_dim >= 0 && (size_t)start_dim < in_shape_rank && 
                    end_dim >= 0 && (size_t)end_dim < in_shape_rank && 
                    start_dim <= end_dim), "invalid start or end dim for node: " << *node);

    std::vector<int64_t> out_shape;

    bool is_dynamic = check_nvtensor_is_dynamic(in);
    nvinfer1::IShuffleLayer* shuffle_layer = engine->network()->addShuffle(*in);
    POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);

    if (is_dynamic) {
        nvinfer1::ITensor* in_shape = engine->network()->addShape(*in)->getOutput(0);
        if (start_dim == end_dim) {
            shuffle_layer->setInput(1, *in_shape);
        } else {
            // Select the dims from start to end with slicelayer and calculate their product.
            // Then, concat the result with other dims to get the new shape.
            std::vector<nvinfer1::ITensor*> cat_nvtensor;
            
            std::vector<int64_t> stride{1};
            std::vector<int64_t> front_start{0}, front_size{start_dim};
            std::vector<int64_t> middle_start{start_dim}, middle_size{end_dim - start_dim + 1};
            std::vector<int64_t> back_start{end_dim + 1}, back_size{(int64_t)in_shape_rank - end_dim - 1};
            
            // front
            if (start_dim > 0) {
                cat_nvtensor.push_back(engine->network()->addSlice(*in_shape,
                                                    sizes_to_nvdim(front_start),
                                                    sizes_to_nvdim(front_size),
                                                    sizes_to_nvdim(stride))->getOutput(0));
            }
            // middle
            nvinfer1::ITensor* middle_tensor = engine->network()->addSlice(*in_shape,
                                                            sizes_to_nvdim(middle_start),
                                                            sizes_to_nvdim(middle_size),
                                                            sizes_to_nvdim(stride))->getOutput(0);
            uint32_t axis_mask = 1;
            // axis_mask |= 1 << 1;
            nvinfer1::IReduceLayer* reduce_prod_layer = engine->network()->addReduce(*middle_tensor, 
                                                            nvinfer1::ReduceOperation::kPROD, axis_mask, true);
            // default is float32, must set int32                                                            
            reduce_prod_layer->setPrecision(nvinfer1::DataType::kINT32);
            
            cat_nvtensor.push_back(reduce_prod_layer->getOutput(0));
            // back
            if ((size_t)end_dim < in_shape_rank - 1) {
                cat_nvtensor.push_back(engine->network()->addSlice(*in_shape,
                                                    sizes_to_nvdim(back_start),
                                                    sizes_to_nvdim(back_size),
                                                    sizes_to_nvdim(stride))->getOutput(0));
            }
            // cat the new shape
            nvinfer1::IConcatenationLayer* concat_layer = 
                                                engine->network()->addConcatenation(cat_nvtensor.data(), cat_nvtensor.size());
            concat_layer->setAxis(0);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
            shuffle_layer->setInput(1, *(concat_layer->getOutput(0)));
        }
    } else {
        // static situation
        out_shape = torch::flatten(torch::rand(in_shape), start_dim, end_dim).sizes().vec();
        shuffle_layer->setReshapeDimensions(sizes_to_nvdim(out_shape));
    }
    
    shuffle_layer->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();
    return true;
}

/**
 * aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
 * aten::view(Tensor(a) self, int[] size) -> Tensor(a)
 * **/
bool PermuteViewConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for PermuteViewConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for PermuteViewConverter is not Tensor as expected");
    //assumes int inputs are all come from prim::Constant.
    // POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
    //     "input[1] for PermuteViewConverter is not come from prim::Constant as expected");
        
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());

    std::vector<int64_t> new_order;
    if (!check_inputs_tensor_scalar(engine, node)) {
        new_order = (engine->context().get_constant(inputs[1])).toIntList().vec();
        LOG(INFO) << "Shuffle to: " << sizes_to_nvdim(new_order);
    }

    auto shuffle = engine->network()->addShuffle(*in);
    POROS_CHECK(shuffle, "Unable to create shuffle layer from node: " << *node);

    if (node->kind() == torch::jit::aten::permute) {
        nvinfer1::Permutation permute;
        std::copy(new_order.begin(), new_order.end(), permute.order);
        shuffle->setSecondTranspose(permute);
    } else if (node->kind() == torch::jit::aten::view) {
        nvinfer1::ITensor* view_size = engine->context().get_tensor(inputs[1]);
        if (view_size != nullptr) {
            shuffle->setInput(1, *view_size);
        } else {
            shuffle->setReshapeDimensions(sizes_to_nvdim(new_order));
        }
    } else {
        POROS_THROW_ERROR("We should never reach here for PermuteViewConverter, meet Unsupported node kind!");
    }

    shuffle->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle->getOutput(0)->getDimensions();
    return true;
}

/**
 * aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
 * **/
bool ReshapeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for ReshapeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ReshapeConverter is not Tensor as expected");
        
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());

    nvinfer1::IShuffleLayer* shuffle_layer = engine->network()->addShuffle(*in);
    POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);

    // 检查是否能使用get_tensor获取input[1]
    if (engine->context().get_tensor(inputs[1]) != nullptr) {
        nvinfer1::ITensor* new_shape = engine->context().get_tensor(inputs[1]);
        shuffle_layer->setInput(1, *new_shape);
    } else {
        std::vector<int64_t> new_order = (engine->context().get_constant(inputs[1])).toIntList().vec();
        // if input shape is dynamic, torch::reshape is wrong.
        // std::vector<int64_t> new_shape = torch::reshape(torch::rand(in_shape), new_order).sizes().vec();
        LOG(INFO) << "Shuffle to: " << sizes_to_nvdim(new_order);
        shuffle_layer->setReshapeDimensions(sizes_to_nvdim(new_order));
    }

    shuffle_layer->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle_layer->getOutput(0)->getDimensions();
    return true;
}

/**
 * aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
 * **/
bool TransposeConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for TransposeConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for TransposeConverter is not Tensor as expected");
    //assumes int inputs are all come from prim::Constant.
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for TransposeConverter is not come from prim::Constant as expected");
    POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for TransposeConverter is not come from prim::Constant as expected");
        
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(in->getDimensions());
    auto ndims = in_shape.size();
    
    //extract dim0 & dim1
    auto dim0 = (engine->context().get_constant(inputs[1])).toInt();
    auto dim1 = (engine->context().get_constant(inputs[2])).toInt();
    
    std::vector<int64_t> new_order;
    for (size_t i = 0; i < ndims; i++) {
        new_order.push_back(i);
    }
    dim0 = dim0 < 0 ? (dim0 + ndims) : dim0;
    dim1 = dim1 < 0 ? (dim1 + ndims) : dim1;
    auto tmp = dim0;
    new_order[dim0] = new_order[dim1];
    new_order[dim1] = tmp;
    LOG(INFO) << "Shuffle to: " << sizes_to_nvdim(new_order);
    
    auto shuffle = engine->network()->addShuffle(*in);
    POROS_CHECK(shuffle, "Unable to create shuffle layer from node: " << *node);
    nvinfer1::Permutation permute;
    std::copy(new_order.begin(), new_order.end(), permute.order);
    shuffle->setSecondTranspose(permute);
    shuffle->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle->getOutput(0)->getDimensions();
    return true;
}

/**
 * aten::t(Tensor(a) self) -> Tensor(a)
 * **/
bool AtenTConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for AtenTConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for AtenTConverter is not Tensor as expected");
   
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto input_dims = in->getDimensions();

    if (input_dims.nbDims < 2) {
        //For aten::t situation. if input tensors < 2D, return them as is 
        engine->context().set_tensor(node->outputs()[0], in);
        LOG(INFO) << "Output tensor shape: " << in->getDimensions();
        return true;
    }
    
    auto shuffle = engine->network()->addShuffle(*in);
    POROS_CHECK(shuffle, "Unable to create shuffle layer from node: " << *node);
    nvinfer1::Permutation first_perm;
    first_perm.order[0] = 1;
    first_perm.order[1] = 0;
    shuffle->setFirstTranspose(first_perm);
    shuffle->setZeroIsPlaceholder(false);
    shuffle->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], shuffle->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << shuffle->getOutput(0)->getDimensions();
    return true;
}

/**
 * aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
 * **/
bool PixelShuffleConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    //basic check
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for PixelShuffleConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for PixelShuffleConverter is not Tensor as expected");
    //assumes int inputs are all come from prim::Constant.
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for PixelShuffleConverter is not come from prim::Constant as expected");

    //extract self
    auto self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);
    auto in_shape = nvdim_to_sizes(self->getDimensions());
    int64_t irank = in_shape.size();
    POROS_CHECK(irank >= 3, "pixel_shuffle expects input to have at least 3 dimensions, but got input with "
                    << std::to_string(irank) << " dimension(s)");
    
    //extract upscale_factor
    int64_t upscale_factor = (engine->context().get_constant(inputs[1])).toInt();
    POROS_CHECK(upscale_factor > 0, "pixel_shuffle expects a positive upscale_factor, but got " 
                    << std::to_string(upscale_factor));
    int64_t upscale_factor_squared = upscale_factor * upscale_factor;


    const auto NUM_NON_BATCH_DIMS = 3;
    const auto self_sizes_batch_end = in_shape.end() - NUM_NON_BATCH_DIMS;

    int64_t ic = in_shape[irank - 3];
    int64_t ih = in_shape[irank - 2];
    int64_t iw = in_shape[irank - 1];
    POROS_CHECK(ic % upscale_factor_squared == 0, 
                    "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
                    << "upscale_factor, but input.size(-3)=" << std::to_string(ic) << " is not divisible by "
                    << std::to_string(upscale_factor_squared));

    int64_t oc = ic / upscale_factor_squared;
    int64_t oh = ih * upscale_factor;
    int64_t ow = iw * upscale_factor;

    std::vector<int64_t> added_dims_shape(in_shape.begin(), self_sizes_batch_end);
    added_dims_shape.insert(added_dims_shape.end(), {oc, upscale_factor, upscale_factor, ih, iw});
    auto view_layer = engine->network()->addShuffle(*self);
    POROS_CHECK(view_layer, "Unable to create shuffle layer from node: " << *node);
    view_layer->setReshapeDimensions(sizes_to_nvdim(added_dims_shape));
    int64_t view_rank = added_dims_shape.size();

    auto permutation_layer = engine->network()->addShuffle(*view_layer->getOutput(0));
    POROS_CHECK(permutation_layer, "Unable to create shuffle layer from node: " << *node);
    std::vector<int64_t> new_order(in_shape.begin(), self_sizes_batch_end);
    std::iota(new_order.begin(), new_order.end(), 0);
    new_order.insert(
        new_order.end(),
        {view_rank - 5, view_rank - 2, view_rank - 4, view_rank - 1, view_rank - 3});
    nvinfer1::Permutation permute;
    std::copy(new_order.begin(), new_order.end(), permute.order);
    permutation_layer->setSecondTranspose(permute);


    std::vector<int64_t> final_shape(in_shape.begin(), self_sizes_batch_end);
    final_shape.insert(final_shape.end(), {oc, oh, ow});
    auto last_view_layer = engine->network()->addShuffle(*permutation_layer->getOutput(0));
    POROS_CHECK(last_view_layer, "Unable to create shuffle layer from node: " << *node);
    last_view_layer->setReshapeDimensions(sizes_to_nvdim(final_shape));
    last_view_layer->setName(layer_info(node).c_str());
    engine->context().set_tensor(node->outputs()[0], last_view_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << last_view_layer->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, FlattenConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, PermuteViewConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, ReshapeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, TransposeConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, AtenTConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, PixelShuffleConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
