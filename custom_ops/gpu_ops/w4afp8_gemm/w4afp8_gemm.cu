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
#include "w4afp8_gemm_kernel.hpp"




template <typename OutputType>
void DisPatchW4AFp8Gemm(
        const phi::dtype::float8_e4m3fn* input,
        const uint8_t* weight,
        const int * tokens,
        const int * tokens_perfix_sum,
        const float * input_row_sum,
        const float * weight_scale,
        OutputType * out,
        const int token_padding_size,
        const int max_tokens,
        const int batch_size,
        const int M,
        const int K,
        cudaStream_t stream) {

    if (M == 7168 && K == 8192 && batch_size == 8 && token_padding_size == 0) {
        w4afp8_gemm<cutlass::float_e4m3_t, OutputType, 7168, 8192, 8, 0>(
            reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
            reinterpret_cast<const cutlass::float_e4m3_t*>(input), 
            out, 
            weight_scale,
            input_row_sum, 
            tokens_perfix_sum, 
            max_tokens, 
            stream);
    } else if (M == 7168 && K == 8192 && batch_size == 8 && token_padding_size == 4096) {
        w4afp8_gemm<cutlass::float_e4m3_t, OutputType, 7168, 8192, 8, 4096>(
            reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
            reinterpret_cast<const cutlass::float_e4m3_t*>(input), 
            out, 
            weight_scale,
            input_row_sum, 
            tokens, 
            max_tokens, 
            stream);
    } else {
        PD_THROW("Not supported shape. M:%d, K:%d, batch_size:%d, token_padding_size:%d\n", M, K, batch_size, token_padding_size);
    }
}

std::vector<paddle::Tensor> W4AFp8Gemm(
        const paddle::Tensor& input,
        const paddle::Tensor& weight,
        const paddle::Tensor& tokens,
        const paddle::Tensor& tokens_perfix_sum,
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
        paddle::Tensor out = paddle::empty({all_tokens, M}, paddle::DataType::BFLOAT16, input.place());
        phi::dtype::bfloat16 *out_data = out.data<phi::dtype::bfloat16>();
        
        if (is_bflot16) {
            DisPatchW4AFp8Gemm(
                input.data<phi::dtype::float8_e4m3fn>(),
                weight.data<uint8_t>(),
                tokens.data<int>(),
                tokens_perfix_sum.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
        } else {
            DisPatchW4AFp8Gemm(
                input.data<phi::dtype::float8_e4m3fn>(),
                weight.data<uint8_t>(),
                tokens.data<int>(),
                tokens_perfix_sum.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::half_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
        }
        
        return {out};
    } else {
        paddle::Tensor out = paddle::empty({batch_size, token_padding_size, M}, paddle::DataType::BFLOAT16, input.place());

        phi::dtype::bfloat16 * out_data = out.data<phi::dtype::bfloat16>();
        
        if (is_bflot16) {
            DisPatchW4AFp8Gemm(
                input.data<phi::dtype::float8_e4m3fn>(),
                weight.data<uint8_t>(),
                tokens.data<int>(),
                tokens_perfix_sum.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
        } else {
            DisPatchW4AFp8Gemm(
                input.data<phi::dtype::float8_e4m3fn>(),
                weight.data<uint8_t>(),
                tokens.data<int>(),
                tokens_perfix_sum.data<int>(),
                input_row_sum.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::half_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
        }

        return {out};
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
             "tokens_perfix_sum",
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
