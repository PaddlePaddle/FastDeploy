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
