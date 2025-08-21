# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

file_dir = "./gpu_ops/wfp8afp8_sparse_gemm/"

gemm_template_head = """
#pragma once
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
"""
gemm_template_case = """
void wfp8afp8_sparse_gemm_M{M}_N{N}_TAILN{TAILN}_K{K}_B{BATCH}_P{PADDING}_{TYPE}(
    const cutlass::float_e4m3_t * weight,
    const uint32_t * sparse_idx,
    const cutlass::float_e4m3_t * input,
    {cutlass_type} * out,
    const float *weight_scale,
    const int *tokens,
    const int max_tokens,
    cudaStream_t stream);
"""

gemm_template_cu_head = """
#include "paddle/extension.h"
#include "wfp8Afp8_sparse_gemm_template.h"
#include "w8a8_sparse_gemm_kernel.hpp"

"""
gemm_template_cu_template = """
void wfp8afp8_sparse_gemm_M{M}_N{N}_TAILN{TAILN}_K{K}_B{BATCH}_P{PADDING}_{TYPE}(
        const cutlass::float_e4m3_t * weight,
        const uint32_t * sparse_idx,
        const cutlass::float_e4m3_t * input,
        {cutlass_type} * out,
        const float *weight_scale,
        const int *tokens,
        const int max_tokens,
        cudaStream_t stream) {{

    constexpr static int M = {M};
    constexpr static int K = {K};
    constexpr static int Batch = {BATCH};
    constexpr static int TokenPackSize = {PADDING};
    constexpr static int kBlockN = {N};
    constexpr static int kBlockN_TAIL = {TAILN};
    constexpr static int kBlockM = 128;
    constexpr static int kBlockK = 128;
    constexpr static int kNWarps = 4 + kBlockM / 16;
    constexpr static int kStages = 5;
    constexpr int kCluster = 1;
    static_assert(K % kBlockK == 0);
    constexpr int kTiles = K / kBlockK;

    using Kernel_traits = Kernel_traits<
        kBlockM, kBlockN, kBlockK, kNWarps, kStages, kTiles,
        M, TokenPackSize, kBlockN_TAIL, kCluster, cutlass::float_e4m3_t,
        {cutlass_type}>;
    run_gemm<cutlass::float_e4m3_t, {cutlass_type},
        Kernel_traits, M, K, Batch, TokenPackSize>
        (weight, sparse_idx, input, out, weight_scale,
        tokens, max_tokens, stream);
}}
"""

gemm_case = [
    [128, 128, 1, 0],
    [7168, 8192, 8, 0],  # eb45T ffn1
]

dtype = ["BF16"]


def get_cutlass_type(type):
    if type == "BF16":
        return "cutlass::bfloat16_t"
    elif type == "FP16":
        return "cutlass::half_t"


template_head_file = open(f"{file_dir}wfp8Afp8_sparse_gemm_template.h", "w")
template_head_file.write(gemm_template_head)

for type in dtype:
    for case in gemm_case:
        for n in range(32, 257, 32):
            template_head_file.write(
                gemm_template_case.format(
                    M=case[0],
                    K=case[1],
                    N=n,
                    BATCH=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    TAILN=0,
                    cutlass_type=get_cutlass_type(type),
                )
            )
            template_head_file.write(
                gemm_template_case.format(
                    M=case[0],
                    K=case[1],
                    N=256,
                    BATCH=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    TAILN=n - 32,
                    cutlass_type=get_cutlass_type(type),
                )
            )

            template_cu_file = open(
                f"{file_dir}wfp8Afp8_sparse_gemm_M{case[0]}_N{n}_TAILN{0}_K{case[1]}_B{case[2]}_P{case[3]}_{type}.cu",
                "w",
            )
            template_cu_file.write(gemm_template_cu_head)
            template_cu_file.write(
                gemm_template_cu_template.format(
                    M=case[0],
                    K=case[1],
                    N=n,
                    BATCH=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    TAILN=0,
                    cutlass_type=get_cutlass_type(type),
                )
            )

            template_cu_file.close()

            template_cu_file = open(
                f"{file_dir}wfp8Afp8_sparse_gemm_M{case[0]}_N{256}_TAILN{n-32}_K{case[1]}_B{case[2]}_P{case[3]}_{type}.cu",
                "w",
            )
            template_cu_file.write(gemm_template_cu_head)
            template_cu_file.write(
                gemm_template_cu_template.format(
                    M=case[0],
                    K=case[1],
                    N=256,
                    BATCH=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    TAILN=n - 32,
                    cutlass_type=get_cutlass_type(type),
                )
            )

            template_cu_file.close()

for type in dtype:
    template_head_file.write("\n")
    template_head_file.write(
        """#define SPARSE_GEMM_SWITCH_{TYPE}(_M, _K, _BATCH, _TokenPaddingSize, _kBlockN, _TailN, ...)  {{        \\
    if (_M == 0 && _K == 0 && _BATCH == 0 && _TokenPaddingSize == 0 && _kBlockN == 0 && _TailN == 0) {{    \\""".format(
            TYPE=type
        )
    )

    template_head_file.write("\n")

    for case in gemm_case:
        for n in range(32, 257, 32):
            template_head_file.write(
                """    }} else if (_M == {M} && _K == {K} && _BATCH == {BATCH} && _TokenPaddingSize == {PADDING} && _kBlockN == {N} && _TailN == {TAILN}) {{                        \\
        wfp8afp8_sparse_gemm_M{M}_N{N}_TAILN{TAILN}_K{K}_B{BATCH}_P{PADDING}_{TYPE}(__VA_ARGS__);  \\""".format(
                    M=case[0], K=case[1], N=n, BATCH=case[2], TYPE=type, PADDING=case[3], TAILN=0
                )
            )
            template_head_file.write("\n")
            template_head_file.write(
                """    }} else if (_M == {M} && _K == {K} && _BATCH == {BATCH} && _TokenPaddingSize == {PADDING} && _kBlockN == {N} && _TailN == {TAILN}) {{                        \\
        wfp8afp8_sparse_gemm_M{M}_N{N}_TAILN{TAILN}_K{K}_B{BATCH}_P{PADDING}_{TYPE}(__VA_ARGS__);  \\""".format(
                    M=case[0], K=case[1], N=256, BATCH=case[2], TYPE=type, PADDING=case[3], TAILN=n - 32
                )
            )
            template_head_file.write("\n")

    template_head_file.write(
        """    } else {   \\
            PADDLE_THROW(phi::errors::Unimplemented("WFp8aFp8 Sparse not supported m=%d k=%d  batch=%d  token_padding_size=%d  kBlockN=%d  tailN=%d\\n", _M, _K, _BATCH, _TokenPaddingSize, _kBlockN, _TailN));    \\
        }     \\
    }"""
    )

template_head_file.close()
