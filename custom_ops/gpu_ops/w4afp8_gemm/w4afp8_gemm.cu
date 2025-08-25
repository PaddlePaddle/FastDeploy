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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#include "helper.h"
#include "paddle/extension.h"
#include "w4afp8_gemm_template.h"


void weight_convert(const uint8_t *weight, uint8_t *weight_new, int batch, int M, int K) {
    assert(K % 64 == 0);
    for (int b = 0; b < batch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; k+=64) {
                for (int k_inner = 0; k_inner < 32; ++k_inner) {
                    uint8_t temp = 0;
                    uint8_t left = weight[b * M * K + m * K + k + k_inner];
                    uint8_t right = weight[b * M * K + m * K + k + k_inner + 32];
                    temp |= left << 4;
                    temp |= right;
                    weight_new[b * M * K / 2 + m * K / 2 + k / 2 + k_inner] = *reinterpret_cast<uint8_t*>(&temp);
                }
            }
        }
    }
}






template <typename OutputType>
void DisPatchW4AFp8Gemm(
        const cutlass::float_e4m3_t* input,
        const cutlass::float_e4m3_t* weight,
        const int * tokens,
        const float * input_row_sum,
        const float * weight_scale,
        OutputType * out,
        const int token_padding_size,
        const int max_tokens,
        const int batch_size,
        const int M,
        const int K,
        cudaStream_t stream) {

    int kBlockN = (max_tokens + 15) / 16 * 16;
    int TailN = 0;
    if (kBlockN > 256) {
        TailN = kBlockN % 256;
        kBlockN = 256;
    }
    if constexpr (std::is_same_v<OutputType, cutlass::bfloat16_t>) {
        GEMM_SWITCH_BF16(
            M, K, batch_size, token_padding_size, kBlockN, TailN,
            weight,
            input,
            out,
            weight_scale,
            input_row_sum,
            tokens,
            max_tokens,
            stream)
    } else {
        PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
}

std::vector<paddle::Tensor> W4AFp8Gemm(
        const paddle::Tensor& input,
        const paddle::Tensor& weight,
        const paddle::Tensor& tokens, // If tokenpadding=0, this tensor represents the prefix sum of tensors, otherwise it represents the number of tokens in each group
        const paddle::Tensor& input_row_sum,
        const paddle::Tensor& weight_scale,
        const int token_padding_size,
        const int max_tokens,
        const bool is_bflot16) {

    const int batch_size = weight.dims()[0];
    const int M = weight.dims()[1];
    const int K = weight.dims()[2] * 2;

    if (input.dtype() != paddle::DataType::FLOAT8_E4M3FN) {
        PD_THROW("Only supported dtype in ['FLOAT8_E4M3FN'].");
    }

    if (token_padding_size == 0) {
        const int all_tokens = input.dims()[0];
        if (is_bflot16) {
            paddle::Tensor out = paddle::empty({all_tokens, M}, paddle::DataType::BFLOAT16, input.place());
            phi::dtype::bfloat16 *out_data = out.data<phi::dtype::bfloat16>();
            DisPatchW4AFp8Gemm(
                reinterpret_cast<const cutlass::float_e4m3_t*>(input.data<phi::dtype::float8_e4m3fn>()),
                reinterpret_cast<const cutlass::float_e4m3_t*>(weight.data<uint8_t>()),
                tokens.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
            return {out};
        } else {
            PD_THROW("Only supported dtype in ['BFLOAT16'].");
        }
    } else {
        if (is_bflot16) {
            paddle::Tensor out = paddle::empty({batch_size, token_padding_size, M}, paddle::DataType::BFLOAT16, input.place());
            phi::dtype::bfloat16 * out_data = out.data<phi::dtype::bfloat16>();
            DisPatchW4AFp8Gemm(
                reinterpret_cast<const cutlass::float_e4m3_t*>(input.data<phi::dtype::float8_e4m3fn>()),
                reinterpret_cast<const cutlass::float_e4m3_t*>(weight.data<uint8_t>()),
                tokens.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
            return {out};
        } else {
            PD_THROW("Only supported dtype in ['BFLOAT16'].");
        }
    }
}


std::vector<paddle::Tensor> W4AFp8GemmWeightConvert(const paddle::Tensor& weight) {
    const int batch_size = weight.dims()[0];
    const int M = weight.dims()[1];
    const int K = weight.dims()[2];
    paddle::Tensor weight_new = paddle::empty({batch_size, M, K / 2}, paddle::DataType::UINT8, weight.place());
    weight_convert(weight.data<uint8_t>(), weight_new.data<uint8_t>(), batch_size, M, K);
    return {weight_new};
}

PD_BUILD_STATIC_OP(w4afp8_gemm)
    .Inputs({"input",
             "weight",
             "tokens",
             "input_row_sum",
             "weight_scale"})
    .Outputs({"out"})
    .Attrs({"token_padding_size: int",
            "max_tokens: int",
            "is_bflot16: bool"})
    .SetKernelFn(PD_KERNEL(W4AFp8Gemm));

PD_BUILD_STATIC_OP(w4afp8_gemm_weight_convert)
    .Inputs({"weight"})
    .Outputs({"converted_weight"})
    .SetKernelFn(PD_KERNEL(W4AFp8GemmWeightConvert));
