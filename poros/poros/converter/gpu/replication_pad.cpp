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
// https://github.com/pytorch/TensorRT/blob/master/core/conversion/converters/impl/replication_pad.cpp
//
// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the 3-Clause BSD License

/**
* @file replication_pad.cpp
* @author tianjinjin@baidu.com
* @date Tue Sep  7 14:29:20 CST 2021
* @brief 
**/

#include "poros/converter/gpu/replication_pad.h"
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
"aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor",
"aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor",
"aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor",
*/
bool ReplicationPadConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 2), "invaid inputs size for ReplicationPadConverter");
    POROS_CHECK_TRUE((inputs[0]->type()->isSubtypeOf(c10::TensorType::get())), 
        "input[0] for ReplicationPadConverter is not Tensor as expected");

    //extract self
    auto in = engine->context().get_tensor(inputs[0]);
    POROS_CHECK_TRUE((in != nullptr), "Unable to init input tensor for node: " << *node);
    auto inDims = in->getDimensions();
    int64_t inRank = inDims.nbDims;

    //extract padding
    auto padding = (engine->context().get_constant(inputs[1])).toIntList().vec();
    if (padding.size() == 1) {
        POROS_THROW_ERROR("Only 3D, 4D, 5D padding with non-constant padding are supported for now");
    }
    if (inRank == 3) {
        POROS_CHECK(padding.size() == 2, "3D tensors expect 2 values for padding");
    } else if (inRank == 4) {
        POROS_CHECK(padding.size() == 4, "4D tensors expect 4 values for padding");
    } else if (inRank == 5) {
        POROS_CHECK(padding.size() == 6, "5D tensors expect 6 values for padding");
    } else {
        POROS_THROW_ERROR("Only 3D, 4D, 5D padding with non-constant padding are supported for now");
    }
    
    std::vector<nvinfer1::ITensor*> tensors_vec;
    // input: (N, C, D_in, H_in, W_in).
    // padding: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    // When axis is inRank - 1, making W_out = W_in + padding_left + padding_right.
    // When axis is inRank - 2, making H_out = H_in + padding_top + padding_bottom.
    // When axis is inRank - 1, making D_out = D_in + padding_front + padding_back.
    for (int64_t i = 0; i < int(padding.size() / 2); i++) {
        int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2, inRank - 3}
        int64_t padding_index = i * 2;

        if (padding[padding_index] > 0) { // left/top/front padding value
            tensors_vec.clear();
            at::Tensor left_indices = torch::tensor({0}, torch::kInt32);
            auto indicesTensor = tensor_to_const(engine, left_indices);
            auto left_gather_layer = engine->network()->addGather(*in, *indicesTensor, axis);
            left_gather_layer->setName((layer_info(node) + "_IGatherLayer_for_left_axis_" + std::to_string(axis)).c_str());
            auto left_gather_out = left_gather_layer->getOutput(0);
            for (int i = 0; i < padding[padding_index]; i++) {
                tensors_vec.push_back(left_gather_out);
            }
            tensors_vec.push_back(in);
            auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
            concat_layer->setAxis(axis);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_left_axis_" + std::to_string(axis)).c_str());
            in = concat_layer->getOutput(0);
            inDims = in->getDimensions();
        }
    
        if (padding[padding_index + 1] > 0) { // right/bottom/back padding value
            tensors_vec.clear();
            tensors_vec.push_back(in);

            nvinfer1::ITensor* indicesTensor = NULL;
            if (inDims.d[axis] == -1) {
                auto shapeTensor = engine->network()->addShape(*in)->getOutput(0);
                at::Tensor dimValue = torch::tensor({axis}, torch::kInt32);
                auto dimTensor = tensor_to_const(engine, dimValue);
                indicesTensor = engine->network()->addGather(*shapeTensor, *dimTensor, 0)->getOutput(0);
                auto oneTensor = tensor_to_const(engine, torch::tensor({1}, torch::kInt32));
                indicesTensor = engine->network()->addElementWise(*indicesTensor, 
                                    *oneTensor, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
            } else {
                auto indices = torch::tensor({inDims.d[axis] - 1}, torch::kInt32);
                indicesTensor = tensor_to_const(engine, indices);
            }
            auto right_gather_layer = engine->network()->addGather(*in, *indicesTensor, axis);
            right_gather_layer->setName((layer_info(node) + "_IGatherLayer_for_right_axis_" + std::to_string(axis)).c_str());
            auto right_gather_out = right_gather_layer->getOutput(0);

            for (int i = 0; i < padding[padding_index + 1]; i++) {
                tensors_vec.push_back(right_gather_out);
            }

            auto concat_layer = engine->network()->addConcatenation(tensors_vec.data(), tensors_vec.size());
            concat_layer->setAxis(axis);
            concat_layer->setName((layer_info(node) + "_IConcatenationLayer_for_right_axis_" + std::to_string(axis)).c_str());
            in = concat_layer->getOutput(0);
            inDims = in->getDimensions();
        }
    }
    
    engine->context().set_tensor(node->outputs()[0], in);
    LOG(INFO) << "Output tensor shape: " << in->getDimensions();
    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, ReplicationPadConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
