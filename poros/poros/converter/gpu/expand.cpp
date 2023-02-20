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
* @file expand.cpp
* @author tianjinjin@baidu.com
* @date Mon Mar  8 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/expand.h"
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
"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"*/
bool ExpandConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3 || inputs.size() == 2), "invaid inputs size for ExpandConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ExpandConverter is not Tensor as expected");
    if (inputs.size() == 3) {
        POROS_CHECK_TRUE((inputs[2]->node()->kind() == torch::jit::prim::Constant),
            "input[2] for ExpandConverter is not come from prim::Constant as expected");
    }
    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto input_dims = in->getDimensions();
    auto input_rank = in->getDimensions().nbDims;

    //extract target_dims & init expanded_dims_tensor
    nvinfer1::Dims target_dims;
    nvinfer1::ITensor* expanded_dims_tensor = nullptr;
    bool is_expand_layer = false;
    bool has_tensor_scalar = false;
    if (node->kind() == torch::jit::aten::expand) {
        has_tensor_scalar = check_inputs_tensor_scalar(engine, node);
        if (has_tensor_scalar) {
            expanded_dims_tensor = get_tensor_scalar(inputs[1]);
            POROS_CHECK_TRUE((expanded_dims_tensor != nullptr), node_info(node) + std::string("get int nvtensor false."));
            target_dims = expanded_dims_tensor->getDimensions();
        } else {
            auto expanded_size = (engine->context().get_constant(inputs[1])).toIntList();
            target_dims = sizes_to_nvdim(expanded_size);
            auto expanded_size_tensor = torch::tensor(expanded_size.vec(), torch::kInt32);
            expanded_dims_tensor = tensor_to_const(engine, expanded_size_tensor);
        }
        is_expand_layer = true;
    } else {  //node->kind() == torch::jit::aten::expand_as
        auto target_tensor = engine->context().get_tensor(inputs[1]);
        target_dims = target_tensor->getDimensions();
        expanded_dims_tensor = engine->network()->addShape(*target_tensor)->getOutput(0);
    }
    auto output_rank = target_dims.nbDims;
    if (has_tensor_scalar) {
        output_rank = target_dims.d[0];
    }
    
    POROS_CHECK(input_rank <= output_rank,
        "Number of dimensions of the desired expansion must be greater than or equal to the number of input dimensions");
    
    auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;

    //situation1: ---------- when input is dynamic shape -------------
    if (is_dynamic_shape) {
        // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
        if (!has_tensor_scalar) {
            for (int64_t i = target_dims.nbDims - 1; i >= 0; --i) {
                int64_t offset = target_dims.nbDims - 1 - i;
                int64_t dim = input_dims.nbDims - 1 - offset;
                int64_t size = (dim >= 0) ? input_dims.d[dim] : 1;
                int64_t target_size = target_dims.d[i];
                // Passing -1 as the size for a dimension means not changing the size of that dimension in expand layer.
                if (target_size != -1) {
                    if (size != target_size) {
                        // if size == -1, we can't validate the expansion before setBindingDimensions.
                        POROS_CHECK_TRUE((size == -1 || size == 1), "The expanded size of tensor (" << std::to_string(target_size) << ")"
                                << " must match the existing size (" << std::to_string(size) << ")" << " at dimension " << i);
                    }
                } else {
                    //expand 的 target_size 不可以出现-1，(因为是intlist)，但expand_as 可以，因为通过shape获取真实的size。
                    POROS_CHECK_TRUE(!(is_expand_layer && dim < 0), "The target dims " << target_dims  << " for node [" 
                                    << node_info(node) << "] is illegal, should not have -1 value");
                }
            }
        } else {
            LOG(INFO) << "aten::expend ints tensor maybe not right, because has no check.";
        }
        
        size_t max_rank = std::max(input_rank, output_rank);
        // Dimensions are right alignment. Eg: an input of [3, 1] and max_rank = 4, the result of concat is [1, 1, 3, 1]
        nvinfer1::ITensor* new_input_shape_tensor = nullptr;
        if (max_rank - input_rank > 0) {
            torch::Tensor the_one = torch::tensor(std::vector<int32_t>(max_rank - input_rank, 1), torch::kInt32);
            auto one_tensor = tensor_to_const(engine, the_one);
            auto in_shape_tensor = engine->network()->addShape(*in)->getOutput(0);
            nvinfer1::ITensor* const args[2] = {one_tensor, in_shape_tensor};
            new_input_shape_tensor =  engine->network()->addConcatenation(args, 2)->getOutput(0);
        } else { //max_rank - input_rank == 0
            new_input_shape_tensor =  engine->network()->addShape(*in)->getOutput(0);
        }
        auto new_output_shape_tensor = expanded_dims_tensor;
        
        // Add a reshape layer to expand dims
        auto shuffle = engine->network()->addShuffle(*in);
        shuffle->setInput(1, *new_input_shape_tensor);
        shuffle->setName((layer_info(node) + "_IShuffleLayer").c_str());
        
        // Start the slicing from beginning of tensor since this is an expand layer
        std::vector<int64_t> start_vec(max_rank, 0);
        nvinfer1::Dims starts_dim = sizes_to_nvdim(c10::IntArrayRef(start_vec));
        at::Tensor th_start = torch::tensor(nvdim_to_sizes(starts_dim), torch::kInt32);
        auto starts = tensor_to_const(engine, th_start);
        
        // compute sizes = max(x,y).
        auto sizes = engine->network()->addElementWise(*new_input_shape_tensor, 
                                            *new_output_shape_tensor, 
                                            nvinfer1::ElementWiseOperation::kMAX)->getOutput(0);
        nvinfer1::Dims sizes_dim{-1, {}};
        sizes_dim.nbDims = max_rank;
        
        // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
        // min(1, sub(input_shape, 1))
        torch::Tensor thOne = torch::tensor({1}, torch::kInt32);
        auto thone_tensor = tensor_to_const(engine, thOne);
        auto x_sub_one = engine->network()->addElementWise(*new_input_shape_tensor,
                                                *thone_tensor,
                                                nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
        auto strides = engine->network()->addElementWise(*thone_tensor,
                                                *x_sub_one,
                                                nvinfer1::ElementWiseOperation::kMIN)->getOutput(0);
        nvinfer1::Dims strides_dim{-1, {}};
        strides_dim.nbDims = max_rank;
        
        // Slice layer does the expansion in TRT. Desired output size is specified by sizes input at index 2.
        auto slice = engine->network()->addSlice(*shuffle->getOutput(0), starts_dim, sizes_dim, strides_dim);
        slice->setInput(1, *starts);
        slice->setInput(2, *sizes);
        slice->setInput(3, *strides);
        slice->setName((layer_info(node) + "_ISliceLayer").c_str());
        
        engine->context().set_tensor(node->outputs()[0], slice->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << slice->getOutput(0)->getDimensions();
        return true;

    //situation2: ---------- when input is NOT dynamic shape -------------    
    } else {
        // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
        for (int64_t i = target_dims.nbDims - 1; i >= 0; --i) {
            int64_t offset = target_dims.nbDims - 1 - i;
            int64_t dim = input_dims.nbDims - 1 - offset;
            int64_t size = (dim >= 0) ? input_dims.d[dim] : 1;
            int64_t target_size = target_dims.d[i];
            // In expand layer passing -1 as the size for a dimension means not changing the size of that dimension.
            if (target_size != -1) {
                if (size != target_size) {
                    POROS_CHECK_TRUE((size == 1), "The expanded size of tensor (" << std::to_string(target_size) << ")"
                            << " must match the existing size (" << std::to_string(size) << ")" << " at dimension " << i);
                }
            } else {
                //target_size 不可以出现 -1.
                POROS_CHECK_TRUE((dim >= 0), "The target dims " << target_dims  << " for node [" 
                                    << node_info(node) << "] is illegal, should not have -1 value");
                // in(3, 1), expand(3, -1, 4) -> expand(3, 3, 4)
                target_dims.d[i] = input_dims.d[dim];
            }
        }

        auto num_expand_dims = target_dims.nbDims - input_dims.nbDims;
        if (num_expand_dims > 0) {
            nvinfer1::Dims reshape_dims;
            reshape_dims.nbDims = target_dims.nbDims;
            for (int64_t i = 0; i < num_expand_dims; i++) {
                reshape_dims.d[i] = 1;
            }
            for (int64_t i = 0; i < input_dims.nbDims; i++) {
                reshape_dims.d[num_expand_dims + i] = input_dims.d[i];
            }
            
            // Add a reshape layer to expand dims
            auto reshape_layer = engine->network()->addShuffle(*in);
            reshape_layer->setReshapeDimensions(reshape_dims);
            reshape_layer->setName((layer_info(node) + "_IShuffleLayer").c_str());
            in = reshape_layer->getOutput(0);
            LOG(INFO) << "Input reshaped to : " << in->getDimensions() << " from " << input_dims;
        }
        
        // Start the slicing from beginning of tensor since this is an expand layer
        std::vector<int64_t> start_vec(target_dims.nbDims, 0);
        auto start_offset = sizes_to_nvdim(c10::IntArrayRef(start_vec));
        
        // Set the stride of non singleton dimension to 1
        std::vector<int64_t> strides_vec(target_dims.nbDims, 0);
        for (int64_t i = 0; i < target_dims.nbDims; i++) {
            strides_vec[i] = (in->getDimensions().d[i] != 1);
        }
        
        auto strides = sizes_to_nvdim(c10::IntArrayRef(strides_vec));
        // Slice layer does the expansion in TRT. Desired output size is specified by target_dims
        auto slice_layer = engine->network()->addSlice(*in, start_offset, target_dims, strides);
        slice_layer->setName((layer_info(node) + "_ISliceLayer").c_str());
        engine->context().set_tensor(node->outputs()[0], slice_layer->getOutput(0));
        LOG(INFO) << "Output tensor shape: " << slice_layer->getOutput(0)->getDimensions();
        return true;
    }
}

/*
"aten::repeat(Tensor self, int[] repeats) -> Tensor",
*/
bool RepeatConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for RepeatConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for RepeatConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[2] for RepeatConverter is not come from prim::Constant as expected");

    //extract in
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto input_dims = in->getDimensions();
    int input_rank = input_dims.nbDims;

    //extract repeats
    auto repeats = (engine->context().get_constant(inputs[1])).toIntList().vec();
    int repeats_rank = repeats.size();
    
    POROS_CHECK(repeats_rank >= input_rank, "Number of repeat dimensions cannot be smaller than number of input dimensions");
    auto num_expand_dims = repeats_rank - input_rank;

    auto is_dynamic_shape = PorosGlobalContext::instance().get_poros_options().is_dynamic;
    if (is_dynamic_shape) {
        nvinfer1::ITensor* new_input_shape_tensor;
        if (num_expand_dims > 0) {
            torch::Tensor the_one = torch::tensor(std::vector<int32_t>(num_expand_dims, 1), torch::kInt32);
            auto one_tensor = tensor_to_const(engine, the_one);
            auto in_shape_tensor = engine->network()->addShape(*in)->getOutput(0);
            nvinfer1::ITensor* const args[2] = {one_tensor, in_shape_tensor};
            new_input_shape_tensor =  engine->network()->addConcatenation(args, 2)->getOutput(0);
        } else { //num_expand_dims == 0
            new_input_shape_tensor =  engine->network()->addShape(*in)->getOutput(0);
        }

        // Add a reshape layer to expand dims
        auto shuffle = engine->network()->addShuffle(*in);
        shuffle->setInput(1, *new_input_shape_tensor);
        shuffle->setName((layer_info(node) + "_IShuffleLayer").c_str());
        in = shuffle->getOutput(0);
    } else {
        if (num_expand_dims > 0) {
            nvinfer1::Dims reshape_dims;
            reshape_dims.nbDims = repeats.size();
            for (int i = 0; i < num_expand_dims; i++) {
                reshape_dims.d[i] = 1;
            }
            for (int i = 0; i < input_rank; i++) {
                reshape_dims.d[num_expand_dims + i] = input_dims.d[i];
            }
            
            // Add a reshape layer to expand dims
            auto reshape_layer = engine->network()->addShuffle(*in);
            reshape_layer->setReshapeDimensions(reshape_dims);
            reshape_layer->setName((layer_info(node) + "_IShuffleLayer").c_str());
            in = reshape_layer->getOutput(0);
            LOG(INFO) << "Input reshaped to : " << in->getDimensions() << " from " << input_dims;
        }
    }

    // Concat across all repeat axes.
    // TODO: Implementation might not be performant. Explore other strategies to improve performance.
    for (int i = repeats.size() - 1; i >= 0; --i) {
        std::vector<nvinfer1::ITensor*> tensors_vec;
        for (int j = 0; j < repeats[i]; j++) {
            tensors_vec.push_back(in);
        }
        auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
        concat_layer->setAxis(i);
        concat_layer->setName((layer_info(node) + "_IConcatenationLayer_" + std::to_string(i)).c_str());
        in = concat_layer->getOutput(0);
    }

    engine->context().set_tensor(node->outputs()[0], in);
    LOG(INFO) << "Output tensor shape: " << in->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ExpandConverter);
POROS_REGISTER_CONVERTER(TensorrtEngine, RepeatConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
