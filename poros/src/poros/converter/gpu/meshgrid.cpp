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
* @file meshgrid.cpp
* @author wangrui39@baidu.com
* @date Monday November 27 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/meshgrid.h"
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

// aten::meshgrid(Tensor[] tensors) -> Tensor[]
bool MeshgridConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 1), "invaid inputs size for MeshgridConverter");

    // 1：获取输入 
    std::vector<nvinfer1::ITensor*> tensorlist;
    //tensorlist.resize(2);
    POROS_CHECK_TRUE((engine->context().get_tensorlist(inputs[0], tensorlist)), "extract tensor list err");

    POROS_CHECK_TRUE((tensorlist.size() == 2), 
        "Expected 2 elements in a tensorlist but found " + std::to_string(tensorlist.size()));
    nvinfer1::ITensor* input1 = tensorlist[0];
    POROS_CHECK_TRUE((input1->getDimensions().nbDims == 1), 
        "Expected scalar or 1D tensor in the tensor list but got " + std::to_string(input1->getDimensions().nbDims));
    nvinfer1::ITensor* input2 = tensorlist[1];
    POROS_CHECK_TRUE((input1->getDimensions().nbDims == 1), 
        "Expected scalar or 1D tensor in the tensor list but got " + std::to_string(input2->getDimensions().nbDims));

    /*std::vector<nvinfer1::ITensor*> output_tensorlist;
    output_tensorlist.emplace_back(input1);
    output_tensorlist.emplace_back(input2);*/

    // 2: 构造返回类型
    nvinfer1::Dims reshape_dim;
    reshape_dim.nbDims = 2;
    reshape_dim.d[0] = 1;
    reshape_dim.d[1] = input1->getDimensions().d[0];
    std::vector<nvinfer1::ITensor*> output_tensorlist;
    output_tensorlist.resize(2);

    // 3：生成return tensorlist[0] unsqueeze + cat + transpose
    // a：unsqueeze
    auto unsqueeze_shuffle_layer1 = engine->network()->addShuffle(*input1);
    POROS_CHECK(unsqueeze_shuffle_layer1, "Unable to create shuffle layer from node: " << *node);
    unsqueeze_shuffle_layer1->setReshapeDimensions(reshape_dim);
    unsqueeze_shuffle_layer1->setName((layer_info(node) + "_IShuffleLayer_for_input1").c_str());
    nvinfer1::ITensor *un_sl_output1 = unsqueeze_shuffle_layer1->getOutput(0);
    
    // b：cat
    std::vector<nvinfer1::ITensor*> cat_tensorlist1;
    cat_tensorlist1.resize(input2->getDimensions().d[0]);
    for (int i = 0; i < input2->getDimensions().d[0]; ++i) {
        auto tmp_weights = Weights(at::zeros({un_sl_output1->getDimensions().d[0], un_sl_output1->getDimensions().d[1]}, {at::kCUDA}).to(torch::kInt));
        auto constant_layer = engine->network()->addConstant(tmp_weights.shape, tmp_weights.data);
        nvinfer1::ITensor* costant_tensor = constant_layer->getOutput(0);
        auto add_layer = engine->network()->addElementWise(*costant_tensor, *un_sl_output1, nvinfer1::ElementWiseOperation::kSUM);
        add_layer->setName((layer_info(node) + "_sum_for_tensorlist1_" + std::to_string(i)).c_str());
        cat_tensorlist1[i] = add_layer->getOutput(0);
    }

    auto cat_layer1 = engine->network()->addConcatenation(cat_tensorlist1.data(), cat_tensorlist1.size());
    cat_layer1->setAxis(0);
    cat_layer1->setName((layer_info(node) + "_IConcatenationLayer_1").c_str());
    nvinfer1::ITensor *cat_output1 = cat_layer1->getOutput(0);

    // c：transpose
    auto transpose_shuffle_layer = engine->network()->addShuffle(*cat_output1);
    POROS_CHECK(transpose_shuffle_layer, "Unable to create shuffle layer from node: " << *node);
    nvinfer1::Permutation permute;
    permute.order[0] = 1;
    permute.order[1] = 0;
    transpose_shuffle_layer->setSecondTranspose(permute);
    transpose_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_cat_output").c_str());
    nvinfer1::ITensor *ts_output = transpose_shuffle_layer->getOutput(0);
    output_tensorlist[0] = ts_output;

    // 4：生成return tensorlist[1] unsqueeze + cat
    // a：unsqueeze
    reshape_dim.d[1] = input2->getDimensions().d[0];
    auto unsqueeze_shuffle_layer2 = engine->network()->addShuffle(*input2);
    POROS_CHECK(unsqueeze_shuffle_layer2, "Unable to create shuffle layer from node: " << *node);
    unsqueeze_shuffle_layer2->setReshapeDimensions(reshape_dim);
    unsqueeze_shuffle_layer2->setName((layer_info(node) + "_IShuffleLayer_for_input2").c_str());
    nvinfer1::ITensor *un_sl_output2 = unsqueeze_shuffle_layer2->getOutput(0);

    // b：cat
    std::vector<nvinfer1::ITensor*> cat_tensorlist2;
    cat_tensorlist2.resize(input1->getDimensions().d[0]);
    for (int i = 0; i < input1->getDimensions().d[0]; ++i) {
        auto tmp_weights = Weights(at::zeros({un_sl_output2->getDimensions().d[0], un_sl_output2->getDimensions().d[1]}, {at::kCUDA}).to(torch::kInt));
        auto constant_layer = engine->network()->addConstant(tmp_weights.shape, tmp_weights.data);
        nvinfer1::ITensor* costant_tensor = constant_layer->getOutput(0);
        auto add_layer = engine->network()->addElementWise(*costant_tensor, *un_sl_output2, nvinfer1::ElementWiseOperation::kSUM);
        //cat_tensorlist2.emplace_back(add_layer->getOutput(0));
        add_layer->setName((layer_info(node) + "_sum_for_tensorlist2_" + std::to_string(i)).c_str());
        cat_tensorlist2[i] = add_layer->getOutput(0);
    }

    auto cat_layer2 = engine->network()->addConcatenation(cat_tensorlist2.data(), cat_tensorlist2.size());
    cat_layer2->setAxis(0);
    cat_layer2->setName((layer_info(node) + "_IConcatenationLayer_2").c_str()); 
    nvinfer1::ITensor *cat_output2 = cat_layer2->getOutput(0);
    output_tensorlist[1] = cat_output2;

    // 5：设置output
    engine->context().set_tensorlist(node->outputs()[0], output_tensorlist);
    return true;
}
  
POROS_REGISTER_CONVERTER(TensorrtEngine, MeshgridConverter);

} // baidu
} // mirana
} // poros
