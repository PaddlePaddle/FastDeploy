// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "wfp8Afp8_sparse_gemm_template.h"

template <typename OutputType>
void DisPatchWFp8AFp8Gemm(
        const cutlass::float_e4m3_t* input,
        const uint32_t* sparse_idx,
        const cutlass::float_e4m3_t* weight,
        const int * tokens,
        const float * weight_scale,
        OutputType * out,
        const int token_padding_size,
        const int max_tokens,
        const int batch_size,
        const int M,
        const int K,
        cudaStream_t stream) {

    const int max_tokens_pack32 = (max_tokens + 31) / 32 * 32;

    int kBlockN = 256;
    int TailN = max_tokens_pack32 % kBlockN;
    if (max_tokens < 256) {
        kBlockN = max_tokens_pack32;
        TailN = 0;
    }
    if constexpr (std::is_same_v<OutputType, cutlass::bfloat16_t>) {
        SPARSE_GEMM_SWITCH_BF16(M, K, batch_size, token_padding_size, kBlockN, TailN,
            weight,
            sparse_idx,
            input,
            out,
            weight_scale,
            tokens,
            max_tokens,
            stream)
    } else {
        PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
}

void WFp8AFp8Gemm(
        const paddle::Tensor& input,
        const paddle::Tensor& sparse_idx,
        const paddle::Tensor& weight,
        const paddle::Tensor& tokens, // If tokenpadding=0, this tensor represents the prefix sum of tensors, otherwise it represents the number of tokens in each group
        const paddle::Tensor& weight_scale,
        const paddle::Tensor& out,
        const int token_padding_size,
        const int max_tokens,
        const bool is_bfloat16) {

    const int batch_size = weight.dims()[0];
    const int M = weight.dims()[1];
    const int K = weight.dims()[2] * 2;

    if (input.dtype() != paddle::DataType::FLOAT8_E4M3FN) {
        PD_THROW("Only supported dtype in ['FLOAT8_E4M3FN'].");
    }

    if (is_bfloat16) {
        DisPatchWFp8AFp8Gemm(
            reinterpret_cast<const cutlass::float_e4m3_t*>(input.data<phi::dtype::float8_e4m3fn>()),
            reinterpret_cast<const uint32_t*>(sparse_idx.data<int32_t>()),
            reinterpret_cast<const cutlass::float_e4m3_t*>(weight.data<phi::dtype::float8_e4m3fn>()),
            tokens.data<int>(),
            weight_scale.data<float>(),
            reinterpret_cast<cutlass::bfloat16_t*>(const_cast<phi::dtype::bfloat16*>(out.data<phi::dtype::bfloat16>())),
            token_padding_size,
            max_tokens,
            batch_size,
            M,
            K,
            input.stream()
        );
    } else {
        PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
}

PD_BUILD_STATIC_OP(wfp8afp8_sparse_gemm)
    .Inputs({"input",
             "sparse_idx",
             "weight",
             "tokens",
             "weight_scale",
             "ffn_out"})
    .Outputs({"out"})
    .SetInplaceMap({{"ffn_out", "out"}})
    .Attrs({"token_padding_size: int",
            "max_tokens: int",
            "is_bfloat16: bool"})
    .SetKernelFn(PD_KERNEL(WFp8AFp8Gemm));
