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
* @file softmax.cpp
* @author tianjinjin@baidu.com
* @date Tue Aug 24 17:15:33 CST 2021
* @brief 
**/

#include "poros/converter/gpu/softmax.h"
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

/*aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor*/
bool SoftmaxConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 3), "invaid inputs size for SoftmaxConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for SoftmaxConverter is not Tensor as expected");
    POROS_CHECK_TRUE((inputs[1]->node()->kind() == torch::jit::prim::Constant), 
        "input[1] for SoftmaxConverter is not come from prim::Constant as expected");
    LOG(INFO) << "Disregarding input[2] dtype argument";

    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto shape = nvdim_to_sizes(in->getDimensions());

    bool is_dynamic = check_nvtensor_is_dynamic(in);
    nvinfer1::ITensor* in_shape_tensor = nullptr;
    if (is_dynamic) {
        in_shape_tensor = engine->network()->addShape(*in)->getOutput(0);
    }
    // SoftMax needs at least 2D input
    if (shape.size() < 2) {
        auto new_shape = sizes_to_nvdim_with_pad(shape, 2);
        auto shuffle = engine->network()->addShuffle(*in);
        shuffle->setReshapeDimensions(new_shape);
        shuffle->setName((layer_info(node) + " [Reshape to " + nvdim_to_str(new_shape) + ']').c_str());
        if (is_dynamic) {
            nvinfer1::ITensor* insert_tensor = tensor_to_const(engine, torch::tensor({1}, torch::kInt32));
            std::vector<nvinfer1::ITensor*> inputs_nvtensor;
            inputs_nvtensor.push_back(insert_tensor);
            inputs_nvtensor.push_back(in_shape_tensor);
            nvinfer1::IConcatenationLayer* concat_layer = 
                    engine->network()->addConcatenation(inputs_nvtensor.data(), inputs_nvtensor.size());
            concat_layer->setAxis(0);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer").c_str());
            nvinfer1::ITensor* concat_out = concat_layer->getOutput(0);
            shuffle->setInput(1, *concat_out);
            shuffle->setName((layer_info(node) + "_IShuffleLayer_1D_to_2D").c_str());
        }
        in = shuffle->getOutput(0);
    }
    
    //extract dim
    auto dim = (engine->context().get_constant(inputs[1])).toInt();
    if (dim < 0) {
        dim = shape.size() + dim;
    }
    
    //main function
    auto softmax = engine->network()->addSoftMax(*in);
    POROS_CHECK(softmax, "Unable to create softmax layer from node: " << *node);
    if (shape.size() > 1) {
        softmax->setAxes(1 << (dim));
    } else {
        // When there is no batch dimension
        softmax->setAxes(1 << (dim + 1));
    }
    softmax->setName((layer_info(node) + "_ISoftMaxLayer").c_str());
    auto out_tensor = softmax->getOutput(0);
    
    // SoftMax reshape back
    if (shape.size() < 2) {
        auto old_shape = sizes_to_nvdim(shape);
        LOG(INFO) << "Input shape was less than 2D got: " << old_shape 
                << ", inserting shuffle layer to reshape back";
        auto shuffle = engine->network()->addShuffle(*out_tensor);
        shuffle->setReshapeDimensions(old_shape);
        shuffle->setName((layer_info(node) + " [Reshape to " + nvdim_to_str(old_shape) + ']').c_str());
        if (is_dynamic) {
            shuffle->setInput(1, *in_shape_tensor);
            shuffle->setName((layer_info(node) + "shuffle_to_old_shape").c_str());
        }
        out_tensor = shuffle->getOutput(0);
    }

    engine->context().set_tensor(node->outputs()[0], out_tensor);
    LOG(INFO) << "Output tensor shape: " << out_tensor->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, SoftmaxConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
