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
* @file roll.cpp
* @author tianshaoqing@baidu.com
* @date Wed Jul 20 16:34:51 CST 2022
* @brief 
**/

#include "poros/converter/gpu/roll.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)
bool RollConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for RollConverter");

    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for RollConverter is not Tensor as expected");

    POROS_CHECK_TRUE((inputs[1]->type()->isSubtypeOf(c10::ListType::ofInts()) 
                        && inputs[2]->type()->isSubtypeOf(c10::ListType::ofInts())), 
                        "input[1] or input[2] for RollConverter is not int[] as expected");
    // extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    // extract shifts
    std::vector<int64_t> shifts_vec = (engine->context().get_constant(inputs[1])).toIntList().vec();
    // extract dims
    std::vector<int64_t> dims_vec = (engine->context().get_constant(inputs[2])).toIntList().vec();

    POROS_CHECK_TRUE((shifts_vec.size() == dims_vec.size()), 
                        "The length of shifts and dims must be equal in RollConverter.");
    
    // Implementation of aten::roll
    // example: 
    // input = {1, 2, 3, 4, 5}; shifts = 3; dim = 0;
    // Then slice input into two parts: {1, 2} and {3, 4, 5}.
    // Finally flip their order and concat them on rolling dim 0: {3, 4, 5, 1, 2}.
    // And so on when multiple dimensions.
    nvinfer1::Dims self_dims = self->getDimensions();
    for (size_t i = 0; i < shifts_vec.size(); i++) {
        std::vector<nvinfer1::ITensor*> tensorlist;
        int64_t rolling_dim = dims_vec[i];
        rolling_dim = (rolling_dim < 0) ? (self_dims.nbDims + rolling_dim) : rolling_dim;

        int64_t shift_stride = shifts_vec[i];
        // Shift is allowed to be greater than the rolling dimension, so we need to take the remainder.
        shift_stride = shift_stride % self_dims.d[rolling_dim];
        // when shift == 0, on processing required
        if (shift_stride == 0) {
            continue;
        }
        std::vector<int64_t> start_vec(self_dims.nbDims, 0);
        std::vector<int64_t> size_vec(self_dims.nbDims, 0);
        std::vector<int64_t> stride_vec(self_dims.nbDims, 1);

        for (int32_t s = 0; s < self_dims.nbDims; s++) {
            size_vec[s] = self_dims.d[s];
        }

        size_vec[rolling_dim] = (shift_stride < 0) ? (-shift_stride) : (self_dims.d[rolling_dim] - shift_stride);
        
        auto slice_left_layer = engine->network()->addSlice(*self, 
                                                sizes_to_nvdim(start_vec), 
                                                sizes_to_nvdim(size_vec), 
                                                sizes_to_nvdim(stride_vec));
        slice_left_layer->setName((layer_info(node) + "_left_slice_" + std::to_string(i)).c_str());
        nvinfer1::ITensor* left_slice = slice_left_layer->getOutput(0);

        start_vec[rolling_dim] = size_vec[rolling_dim];
        size_vec[rolling_dim] = self_dims.d[rolling_dim] - size_vec[rolling_dim];

        auto slice_right_layer = engine->network()->addSlice(*self, 
                                                sizes_to_nvdim(start_vec), 
                                                sizes_to_nvdim(size_vec), 
                                                sizes_to_nvdim(stride_vec));
        slice_right_layer->setName((layer_info(node) + "_right_slice_" + std::to_string(i)).c_str());
        nvinfer1::ITensor* right_slice = slice_right_layer->getOutput(0);
        tensorlist.push_back(right_slice);
        tensorlist.push_back(left_slice);

        auto cat_layer = engine->network()->addConcatenation(tensorlist.data(), tensorlist.size());
        cat_layer->setAxis(static_cast<int>(rolling_dim));
        cat_layer->setName((layer_info(node) + "_cat_" + std::to_string(i)).c_str());
        self = cat_layer->getOutput(0);
    }

    engine->context().set_tensor(node->outputs()[0], self);
    LOG(INFO) << "Output shape: " << self->getDimensions();    
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, RollConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
