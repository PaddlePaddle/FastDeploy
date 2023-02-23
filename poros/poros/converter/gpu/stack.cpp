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
* @file stack.cpp
* @author tianjinjin@baidu.com
* @date Tue Sep  7 15:09:14 CST 2021
* @brief 
**/

#include "poros/converter/gpu/stack.h"
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
"aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
"aten::vstack(Tensor[] tensors) -> Tensor"
*/

bool StackConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1 || inputs.size() == 2), "invaid inputs size for StackConverter");
    POROS_CHECK_TRUE(inputs[0]->type()->isSubtypeOf(c10::ListType::ofTensors()), 
        "input[0] for StackConverter is not TensorList as expected");

    //extract tensors
    std::vector<nvinfer1::ITensor*> tensorlist;
    POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensorlist error");

    int64_t dim = 0;
    
    std::vector<nvinfer1::ITensor*> tensors;
    if (inputs.size() == 2) {
        // aten::stack
        POROS_CHECK_TRUE(inputs[1]->type()->isSubtypeOf(c10::NumberType::get()), 
            "input[1] for StackConverter is not int64_t as expected");

        //extract dims
        dim = (engine->context().get_constant(inputs[1])).toInt();
        
        // aten::stack should unsqueeze dims
        // check if input tensorlist is dynamic.
        bool is_dynamic = check_nvtensor_is_dynamic(tensorlist[0]);
        nvinfer1::Dims inputs_dims = tensorlist[0]->getDimensions();

        // when dim is negtive
        if (dim < 0) {
            dim = inputs_dims.nbDims + dim + 1;
        }
        // generate unsqueeze dimensions by shapetensor if dynamic.
        nvinfer1::ITensor* unsqueeze_dim = nullptr;
        if (is_dynamic) {
            nvinfer1::ITensor* input_shapetensor = engine->network()->addShape(*(tensorlist[0]))->getOutput(0);
            unsqueeze_dim = unsqueeze_nv_shapetensor(engine, input_shapetensor, dim);
            if (unsqueeze_dim == nullptr) {
                LOG(INFO) << "unsqueeze nv shape tensor failed";
                return false;
            }
        }
        // unsqueeze each tensor in tensorlist
        for (size_t i = 0; i < tensorlist.size(); ++i) {
            auto shuffle_layer = engine->network()->addShuffle(*tensorlist[i]);
            POROS_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *node);
            if (is_dynamic) {
                shuffle_layer->setInput(1, *unsqueeze_dim);
            } else {
                shuffle_layer->setReshapeDimensions(unsqueeze_dims(tensorlist[i]->getDimensions(), dim));
            }
            shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_tensor_" + std::to_string(i)).c_str());
            tensors.push_back(shuffle_layer->getOutput(0));
        }
    } else {
        // aten::vstack need not unsqueeze dims
        tensors = tensorlist;
    }

    auto cat_layer = engine->network()->addConcatenation(tensors.data(), tensors.size());
    POROS_CHECK(cat_layer, "Unable to create concatenation layer from node: " << *node);
    cat_layer->setAxis(static_cast<int>(dim));
    cat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
    engine->context().set_tensor(node->outputs()[0], cat_layer->getOutput(0));
    LOG(INFO) << "Output tensor shape: " << cat_layer->getOutput(0)->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, StackConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
