// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <vector>
#include "helper.h"

std::vector<paddle::Tensor> W4AFp8Gemm(
    const paddle::Tensor& input,
    const paddle::Tensor& weight,
    const paddle::Tensor&
        tokens,  // If tokenpadding=0, this tensor represents the prefix sum of
                 // tensors, otherwise it represents the number of tokens in
                 // each group
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& input_dequant_scale,
    const int64_t token_padding_size,
    const int64_t max_tokens,
    const bool is_bfloat16);

template <typename InputType, typename OutputType>
void DisPatchW4AFp8GemmWrapper(const InputType* input,
                               const InputType* weight,
                               const int64_t* tokens,
                               const float* input_dequant_scale,
                               const float* weight_scale,
                               OutputType* out,
                               const int64_t token_padding_size,
                               const int64_t max_tokens,
                               const int num_experts,
                               const int64_t M,
                               const int64_t K,
                               const int WeightScaleGroup,
                               cudaStream_t stream);
