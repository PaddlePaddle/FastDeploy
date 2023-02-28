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
* @file lstm.cpp
* @author wangrui39@baidu.com
* @date Mon December 13 11:36:11 CST 2021
* @brief 
**/

#include "poros/converter/gpu/lstm.h"
#include "poros/converter/gpu/weight.h"
#include "poros/converter/gpu/converter_util.h"
#include "poros/engine/tensorrt_engine.h"
#include "poros/engine/trtengine_util.h"
#include "poros/context/poros_global.h"
#include "poros/util/macros.h"
#include "poros/util/poros_util.h"
#include "poros/converter/gpu/add.h"

namespace baidu {
namespace mirana {
namespace poros {

bool add_rnnv2_params(at::Tensor params, nvinfer1::IRNNv2Layer* &layer, bool isW, int rela_index, 
                        int hidden_size, int idx, nvinfer1::RNNGateType gate, bool bias = false) {
    std::vector<at::Tensor> w;
    for (int i = idx * hidden_size; i < hidden_size * (idx + 1); i++){
        w.push_back(params[i].unsqueeze(0));
    }
    if (bias) {
        layer->setBiasForGate(rela_index, gate, isW, Weights(at::cat(w, 0)).data);
    }
    else {
        layer->setWeightsForGate(rela_index, gate, isW, Weights(at::cat(w, 0)).data);
    }
    return true;
}

bool LstmConverter::converter(TensorrtEngine* engine, const torch::jit::Node *node) {
    /*aten::lstm.input(Tensor input,
                       Tensor[] hx,
                       Tensor[] params,
                       bool has_biases,
                       int num_layers,
                       float dropout,
                       bool train,
                       bool bidirectional,
                       bool batch_first) -> (Tensor, Tensor, Tensor))*/
    at::ArrayRef<const torch::jit::Value*> inputs = node->inputs();
    POROS_CHECK_TRUE((inputs.size() == 9), "invaid inputs size for LstmConverter");

    // 获取输入
    nvinfer1::ITensor *input = engine->context().get_tensor(inputs[0]);
    std::vector<nvinfer1::ITensor*> hx_tensorlist;
    engine->context().get_tensorlist(inputs[1], hx_tensorlist);
    POROS_CHECK_TRUE((hx_tensorlist.size() == 2), "Unable to init input List[tensor] for node: " << *node);

    // 获取参数
    torch::jit::IValue params = engine->context().get_constant(inputs[2]);
    POROS_CHECK_TRUE((params.isTensorList()), "Unable to init second input List[tensor] for node: " << *node);
    c10::List<at::Tensor> param_list =  params.toTensorList();
    int num_layers = engine->context().get_constant(inputs[4]).toInt();
    bool bidirectional = engine->context().get_constant(inputs[7]).toBool();
    bool batch_first = engine->context().get_constant(inputs[8]).toBool();

    // 获取构建trt rnnlayer的输入
    nvinfer1::ITensor *h_0 = hx_tensorlist[0];
    nvinfer1::ITensor *c_0 = hx_tensorlist[1];
    int32_t hidden_size = c_0->getDimensions().d[c_0->getDimensions().nbDims - 1];

    if (!batch_first) {
        auto input_shuffle_layer = engine->network()->addShuffle(*input);
        input_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
        input_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_input").c_str());
        input = input_shuffle_layer->getOutput(0);
    }
    int max_seqlen = input->getDimensions().d[1];

    // 使用trt现有的lstm
    auto rnnv2_layer = engine->network()->addRNNv2(*input, num_layers, hidden_size, max_seqlen, nvinfer1::RNNOperation::kLSTM);
    if (bidirectional) {
        rnnv2_layer->setDirection(nvinfer1::RNNDirection::kBIDIRECTION);
    }
    rnnv2_layer->setName((layer_info(node) + "_IRNNv2Layer").c_str());

    auto c_0_shuffle_layer = engine->network()->addShuffle(*c_0);
    c_0_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
    c_0_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_c0").c_str());
    rnnv2_layer->setCellState(*c_0_shuffle_layer->getOutput(0));

    auto h_0_shuffle_layer = engine->network()->addShuffle(*h_0);
    h_0_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
    h_0_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_h0").c_str());
    rnnv2_layer->setHiddenState(*h_0_shuffle_layer->getOutput(0));

    // 循环生成layer
    for (int i = 0; i < num_layers; i++){
        size_t rela_index = 0;
        if (bidirectional) {
            rela_index = 2 * i;
        }
        else {
            rela_index = i;
        }
        
        // weight_ih_l
        add_rnnv2_params(param_list[rela_index * 4 + 0], rnnv2_layer, true, rela_index, hidden_size, 0, nvinfer1::RNNGateType::kINPUT);
        add_rnnv2_params(param_list[rela_index * 4 + 0], rnnv2_layer, true, rela_index, hidden_size, 1, nvinfer1::RNNGateType::kFORGET);
        add_rnnv2_params(param_list[rela_index * 4 + 0], rnnv2_layer, true, rela_index, hidden_size, 2, nvinfer1::RNNGateType::kCELL);
        add_rnnv2_params(param_list[rela_index * 4 + 0], rnnv2_layer, true, rela_index, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT);

        // weight_hh_l
        add_rnnv2_params(param_list[rela_index * 4 + 1], rnnv2_layer, false, rela_index, hidden_size, 0, nvinfer1::RNNGateType::kINPUT);
        add_rnnv2_params(param_list[rela_index * 4 + 1], rnnv2_layer, false, rela_index, hidden_size, 1, nvinfer1::RNNGateType::kFORGET);
        add_rnnv2_params(param_list[rela_index * 4 + 1], rnnv2_layer, false, rela_index, hidden_size, 2, nvinfer1::RNNGateType::kCELL);
        add_rnnv2_params(param_list[rela_index * 4 + 1], rnnv2_layer, false, rela_index, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT);

        // bias_ih_l
        add_rnnv2_params(param_list[rela_index * 4 + 2], rnnv2_layer, true, rela_index, hidden_size, 0, nvinfer1::RNNGateType::kINPUT, true);
        add_rnnv2_params(param_list[rela_index * 4 + 2], rnnv2_layer, true, rela_index, hidden_size, 1, nvinfer1::RNNGateType::kFORGET, true);
        add_rnnv2_params(param_list[rela_index * 4 + 2], rnnv2_layer, true, rela_index, hidden_size, 2, nvinfer1::RNNGateType::kCELL, true);
        add_rnnv2_params(param_list[rela_index * 4 + 2], rnnv2_layer, true, rela_index, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT, true);

        // bias_hh_l
        add_rnnv2_params(param_list[rela_index * 4 + 3], rnnv2_layer, false, rela_index, hidden_size, 0, nvinfer1::RNNGateType::kINPUT, true);
        add_rnnv2_params(param_list[rela_index * 4 + 3], rnnv2_layer, false, rela_index, hidden_size, 1, nvinfer1::RNNGateType::kFORGET, true);
        add_rnnv2_params(param_list[rela_index * 4 + 3], rnnv2_layer, false, rela_index, hidden_size, 2, nvinfer1::RNNGateType::kCELL, true);
        add_rnnv2_params(param_list[rela_index * 4 + 3], rnnv2_layer, false, rela_index, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT, true);

        if (bidirectional) {
            // ================reverse=====================
            // weight_ih_l
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 0], rnnv2_layer, true, rela_index + 1, hidden_size, 0, nvinfer1::RNNGateType::kINPUT);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 0], rnnv2_layer, true, rela_index + 1, hidden_size, 1, nvinfer1::RNNGateType::kFORGET);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 0], rnnv2_layer, true, rela_index + 1, hidden_size, 2, nvinfer1::RNNGateType::kCELL);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 0], rnnv2_layer, true, rela_index + 1, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT);

            // weight_hh_l
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 1], rnnv2_layer, false, rela_index + 1, hidden_size, 0, nvinfer1::RNNGateType::kINPUT);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 1], rnnv2_layer, false, rela_index + 1, hidden_size, 1, nvinfer1::RNNGateType::kFORGET);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 1], rnnv2_layer, false, rela_index + 1, hidden_size, 2, nvinfer1::RNNGateType::kCELL);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 1], rnnv2_layer, false, rela_index + 1, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT);

            // bias_ih_l
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 2], rnnv2_layer, true, rela_index + 1, hidden_size, 0, nvinfer1::RNNGateType::kINPUT, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 2], rnnv2_layer, true, rela_index + 1, hidden_size, 1, nvinfer1::RNNGateType::kFORGET, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 2], rnnv2_layer, true, rela_index + 1, hidden_size, 2, nvinfer1::RNNGateType::kCELL, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 2], rnnv2_layer, true, rela_index + 1, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT, true);

            // bias_hh_l
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 3], rnnv2_layer, false, rela_index + 1, hidden_size, 0, nvinfer1::RNNGateType::kINPUT, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 3], rnnv2_layer, false, rela_index + 1, hidden_size, 1, nvinfer1::RNNGateType::kFORGET, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 3], rnnv2_layer, false, rela_index + 1, hidden_size, 2, nvinfer1::RNNGateType::kCELL, true);
            add_rnnv2_params(param_list[(rela_index + 1) * 4 + 3], rnnv2_layer, false, rela_index + 1, hidden_size, 3, nvinfer1::RNNGateType::kOUTPUT, true);
        }
    }

    nvinfer1::ITensor* output = rnnv2_layer->getOutput(0);
    if (!batch_first) {
        auto output1_shuffle_layer = engine->network()->addShuffle(*output);
        output1_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
        output1_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_output").c_str());
        output = output1_shuffle_layer->getOutput(0);
    }
    auto output2_shuffle_layer = engine->network()->addShuffle(*rnnv2_layer->getOutput(1));
    output2_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
    output2_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_output1").c_str());
    auto output3_shuffle_layer = engine->network()->addShuffle(*rnnv2_layer->getOutput(2));
    output3_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
    output3_shuffle_layer->setName((layer_info(node) + "_IShuffleLayer_for_output2").c_str());

    engine->context().set_tensor(node->outputs()[0], output);
    engine->context().set_tensor(node->outputs()[1], output2_shuffle_layer->getOutput(0));
    engine->context().set_tensor(node->outputs()[2], output3_shuffle_layer->getOutput(0));

    return true;
}

POROS_REGISTER_CONVERTER(TensorrtEngine, LstmConverter);

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
