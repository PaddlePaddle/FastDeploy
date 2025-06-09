// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "layers_decoder.h"
#include "paddle/extension.h"
#include "paddle/phi/core/kernel_registry.h"

std::vector<paddle::Tensor> InvokeAllLLaMALayer(
    const paddle::Tensor &input,
    const std::vector<paddle::Tensor> &ln1Gamma,
    const std::vector<paddle::Tensor> &ln1Beta,
    const std::vector<paddle::Tensor> &qkvWeight,
    const std::vector<paddle::Tensor> &qkvBiasWeight,
    const std::vector<paddle::Tensor> &attnOutWeight,
    const std::vector<paddle::Tensor> &attnOutBias,
    const std::vector<paddle::Tensor> &ln2Gamma,
    const std::vector<paddle::Tensor> &ln2Beta,
    const std::vector<paddle::Tensor> &gateWeight,
    const std::vector<paddle::Tensor> &gateBias,
    const std::vector<paddle::Tensor> &upWeight,
    const std::vector<paddle::Tensor> &upBias,
    const std::vector<paddle::Tensor> &downWeight,
    const std::vector<paddle::Tensor> &downBias,
    const paddle::Tensor &pastSeqLen,
    const paddle::Tensor &currentSeqLen,
    const paddle::Tensor &step,
    int hiddensize,
    int totalLayer,
    const std::string &computeType,
    const std::string &activation,
    const std::string &normType,
    int attHeadDim,
    int attHeadNum,
    int kvHeadNum,
    int maxPositions,
    int maxPosEmbed,
    int intermediateSize) {
    auto out = paddle::empty_like(input);
    auto batchSize = input.shape()[0];
    auto inputSeqLen = input.shape()[1];
    auto past_seq_len = pastSeqLen.data<int64_t>()[0];
    auto cur_seq_len = static_cast<int64_t>(currentSeqLen.data<int32_t>()[0]);
    auto step_id = step.data<int64_t>()[0];
    auto output_ptr = reinterpret_cast<void *>(out.data<float>());
    auto xft_data_type = xft::DataType::fp16;
    if (computeType == "bf16") {
        xft_data_type = xft::DataType::bf16;
    } else if (computeType == "bf16_int8") {
        xft_data_type = xft::DataType::bf16_int8;
    }
    auto xft_act_type = xft::ActivationType::SILU;
    if (activation == "relu") {
        xft_act_type = xft::ActivationType::RELU;
    } else if (activation == "gelu") {
        xft_act_type = xft::ActivationType::GELU;
    } else if (activation == "swiglu") {
        xft_act_type = xft::ActivationType::SWIGLU;
    }
    auto xft_norm_type = xft::NormType::RMS;
    if (normType == "layernorm") {
        xft_norm_type = xft::NormType::LN;
    }
    auto input_ptr = reinterpret_cast<const void *>(input.data<float>());
    for (int i = 0; i < totalLayer; ++i) {
        auto ln1Gamma_ptr =
            reinterpret_cast<const float *>(ln1Gamma[i].data<float>());
        auto ln1Beta_ptr =
            reinterpret_cast<const float *>(ln1Beta[i].data<float>());
        auto qkvWeight_ptr =
            reinterpret_cast<const void *>(qkvWeight[i].data<float>());
        auto qkvBiasWeight_ptr =
            reinterpret_cast<const float *>(qkvBiasWeight[i].data<float>());
        auto attnOutWeight_ptr =
            reinterpret_cast<const void *>(attnOutWeight[i].data<float>());
        auto ln2Gamma_ptr =
            reinterpret_cast<const float *>(ln2Gamma[i].data<float>());
        auto ln2Beta_ptr =
            reinterpret_cast<const float *>(ln2Beta[i].data<float>());
        auto gate_weight_ptr =
            reinterpret_cast<const void *>(gateWeight[i].data<float>());
        auto up_weight_ptr =
            reinterpret_cast<const void *>(upWeight[i].data<float>());
        auto down_weight_ptr =
            reinterpret_cast<const void *>(downWeight[i].data<float>());
        auto gate_bias_ptr =
            reinterpret_cast<const float *>(gateBias[i].data<float>());
        auto up_bias_ptr =
            reinterpret_cast<const float *>(upBias[i].data<float>());
        auto down_bias_ptr =
            reinterpret_cast<const float *>(downBias[i].data<float>());
        auto attnOutBias_ptr =
            reinterpret_cast<const float *>(attnOutBias[i].data<float>());
        invokeLayerLLaMA(
            xft_data_type,                         // dt
            xft_act_type,                          // at
            xft_norm_type,                         // nt
            i,                                     // layerId
            totalLayer,                            // totalLayers
            batchSize,                             // batchSize
            inputSeqLen,                           // inputSeqLen
            attHeadDim,                            // attHeadDim
            attHeadNum,                            // attHeadNum
            kvHeadNum,                             // kvHeadNum
            maxPositions,                          // maxPositions
            maxPosEmbed,                           // maxPosEmbed
            past_seq_len,                          // pastSeqLen
            cur_seq_len,                           // currentSeqLen
            step_id,                               // step
            hiddensize,                            // hiddenSize
            intermediateSize,                      // intermediateSize
            reinterpret_cast<void *>(output_ptr),  // output
            hiddensize,                            // outputStride
            input_ptr,                             // input
            hiddensize,                            // inputStride
            ln1Gamma_ptr,                          // ln1Gamma
            ln1Beta_ptr,                           // ln1Beta
            qkvWeight_ptr,                         // queryWeight
            qkvWeight_ptr + hiddensize,            // keyWeight
            qkvWeight_ptr + hiddensize + kvHeadNum * attHeadDim,  // valueWeight
            attnOutWeight_ptr,  // attnOutWeight
            ln2Gamma_ptr,       // ln2Gamma
            ln2Beta_ptr,        // ln2Beta
            gate_weight_ptr,
            up_weight_ptr,
            down_weight_ptr,
            qkvBiasWeight_ptr,               // queryBias
            qkvBiasWeight_ptr + hiddensize,  // keyBias
            qkvBiasWeight_ptr + hiddensize +
                kvHeadNum * attHeadDim,  // valueBias
            attnOutBias_ptr,             // attnOutBias
            qkvWeight_ptr,               // myqkvWeight
            gate_bias_ptr,
            up_bias_ptr,
            down_bias_ptr,
            qkvBiasWeight_ptr);
        if (i < totalLayer - 1) {
            memcpy(const_cast<void *>(input_ptr),
                   output_ptr,
                   batchSize * inputSeqLen * hiddensize * sizeof(float));
        }
    }
    return {out};
}

std::vector<std::vector<int64_t>> AllLLaMALayerInferShape(
    std::vector<int64_t> x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> AllLLaMALayerInferDtype(
    paddle::DataType x_dtype) {
    return {x_dtype};
}

PD_BUILD_STATIC_OP(xft_llama_all_layer)
    .Inputs({
        "x",
        paddle::Vec("ln1Gamma"),
        paddle::Vec("ln1Beta"),
        paddle::Vec("qkvWeight"),
        paddle::Vec("qkvBiasWeight"),
        paddle::Vec("attnOutWeight"),
        paddle::Vec("attnOutBias"),
        paddle::Vec("ln2Gamma"),
        paddle::Vec("ln2Beta"),
        paddle::Vec("gateWeight"),
        paddle::Vec("gateBias"),
        paddle::Vec("upWeight"),
        paddle::Vec("upBias"),
        paddle::Vec("downWeight"),
        paddle::Vec("downBias"),
        "pastSeqLen",
        "currentSeqLen",
        "step",
    })
    .Outputs({"out"})
    .Attrs({"hiddensize :int",
            "totalLayer :int",
            "computeType : std::string",
            "activation :std::string",
            "normType :std::string",
            "attHeadDim: int",
            "attHeadNum: int",
            "kvHeadNum: int",
            "maxPositions: int",
            "maxPosEmbed: int",
            "intermediateSize: int"})
    .SetKernelFn(PD_KERNEL(InvokeAllLLaMALayer))
    .SetInferShapeFn(PD_INFER_SHAPE(AllLLaMALayerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AllLLaMALayerInferDtype));
