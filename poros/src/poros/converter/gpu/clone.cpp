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
* @file clone.cpp
* @author tianshaoqing@baidu.com
* @date Tue Nov 23 12:26:28 CST 2021
* @brief 
**/

#include "poros/converter/gpu/clone.h"
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

// aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
bool CloneConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for CloneConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for CloneConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant),
        "input[1] for CloneConverter is not come from prim::Constant as expected");
    
    POROS_CHECK_TRUE(engine->context().get_constant(inputs[1]).isNone(),
        "not support memory format set yet.");

    //extract self
    nvinfer1::ITensor* self = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((self != nullptr), "Unable to init input tensor for node: " << *node);

    // select whole input tensor to clone a new tensor
    nvinfer1::Dims self_dims = self->getDimensions();
    bool is_dynamic = check_nvtensor_is_dynamic(self);
    
    std::vector<int64_t> start_vec, size_vec, stride_vec;
    for (int32_t i = 0; i < self_dims.nbDims; i++) {
        start_vec.push_back(0);
        if (is_dynamic) {
            size_vec.push_back(0);
        } else {
            size_vec.push_back(self_dims.d[i]);
        }
        stride_vec.push_back(1);
    }

    nvinfer1::Dims start_dim = sizes_to_nvdim(start_vec);
    nvinfer1::Dims size_dim = sizes_to_nvdim(size_vec);
    nvinfer1::Dims stride_dim = sizes_to_nvdim(stride_vec);

    nvinfer1::ITensor* self_shape = nullptr;
    if (is_dynamic) {
        self_shape = engine->network()->addShape(*self)->getOutput(0);
    }
    
    nvinfer1::ISliceLayer* slice_layer = engine->network()->addSlice(*self, start_dim, size_dim, stride_dim);
    POROS_CHECK(slice_layer, "Unable to create slice layer from node: " << *node);
    if (is_dynamic) {
        slice_layer->setInput(2, *self_shape);
    }
    slice_layer->setName((layer_info(node) + "_ISliceLayer").c_str());
    nvinfer1::ITensor* output = slice_layer->getOutput(0);
    
    engine->context().set_tensor(node->outputs()[0], self);
    LOG(INFO) << "Output tensor shape: " << output->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, CloneConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu