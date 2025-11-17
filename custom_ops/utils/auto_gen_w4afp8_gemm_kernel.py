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
import os
import re

file_dir = "./gpu_ops/w4afp8_gemm/"

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
void w4afp8_gemm_M{M}_N{N}_G{GROUPSIZE}_K{K}_E{EXPERTS}_P{PADDING}_{TYPE}(
    const cutlass::float_e4m3_t * weight,
    const cutlass::float_e4m3_t * input,
    {cutlass_type} * out,
    const float *weight_scale,
    const float * input_dequant_scale,
    const int64_t *tokens,
    const int64_t max_tokens,
    cudaStream_t stream);
"""

gemm_template_cu_head = """
#include "paddle/extension.h"
#include "w4afp8_gemm_template.h"
#include "w4afp8_gemm_kernel.hpp"

"""
gemm_template_cu_template = """
void w4afp8_gemm_M{M}_N{N}_G{GROUPSIZE}_K{K}_E{EXPERTS}_P{PADDING}_{TYPE}(
        const cutlass::float_e4m3_t * weight,
        const cutlass::float_e4m3_t * input,
        {cutlass_type} * out,
        const float *weight_scale,
        const float * input_dequant_scale,
        const int64_t *tokens,
        const int64_t max_tokens,
        cudaStream_t stream) {{

    constexpr static int M = {M};
    constexpr static int K = {K};
    constexpr static int EXPERTS = {EXPERTS};
    constexpr static int TokenPackSize = {PADDING};
    constexpr static int kBlockN = {N};
    constexpr static int kGroupSize = {GROUPSIZE};
    constexpr static int kBlockM = 128;
    constexpr static int kBlockK = 128;
    constexpr static int kNWarps = 4 + kBlockM / 16;
    constexpr static int kStages = 5;
    constexpr int kCluster = 1;
    static_assert(K % kBlockK == 0);
    constexpr int kTiles = K / kBlockK;

    using Kernel_traits = Kernel_traits<
        kBlockM, kBlockN, kBlockK, kNWarps, kStages, kTiles,
        M, K, TokenPackSize, kGroupSize, kCluster, cutlass::float_e4m3_t,
        {cutlass_type}>;
    run_gemm<cutlass::float_e4m3_t, {cutlass_type},
        Kernel_traits, M, K, EXPERTS, TokenPackSize, kGroupSize>
        (weight, input, out, weight_scale, input_dequant_scale, tokens, max_tokens, stream);
}}
"""

# [M, K, Number of experts, token Padding Size, weight K group size]
gemm_case = [[256, 256, 2, 0, 128]]

dtype = ["BF16"]

use_fast_compile = True
n_range = [256] if use_fast_compile else [i for i in range(16, 257, 16)]

all_cu_files = []
for type in dtype:
    for case in gemm_case:
        for n in n_range:
            all_cu_files.append(f"w4afp8_gemm_M{case[0]}_N{n}_G{case[4]}_K{case[1]}_E{case[2]}_P{case[3]}_{type}.cu")

for file_path, empty_list, file_name_list in os.walk(file_dir):
    for file_name in file_name_list:
        if re.match(r"^w4afp8_gemm_M\d+_N\d+_.*\.cu$", file_name):
            if file_name not in all_cu_files:
                print("delete w4afp8 kernel file", file_path + file_name)
                os.remove(file_path + file_name)


def get_cutlass_type(type):
    if type == "BF16":
        return "cutlass::bfloat16_t"
    elif type == "FP16":
        return "cutlass::half_t"


template_head_file = open(f"{file_dir}w4afp8_gemm_template.h", "w")
template_head_file.write(gemm_template_head)

for type in dtype:
    for case in gemm_case:
        for n in n_range:
            template_head_file.write(
                gemm_template_case.format(
                    M=case[0],
                    K=case[1],
                    N=n,
                    EXPERTS=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    GROUPSIZE=case[4],
                    cutlass_type=get_cutlass_type(type),
                )
            )

            template_cu_file = open(
                f"{file_dir}w4afp8_gemm_M{case[0]}_N{n}_G{case[4]}_K{case[1]}_E{case[2]}_P{case[3]}_{type}.cu", "w"
            )
            template_cu_file.write(gemm_template_cu_head)
            template_cu_file.write(
                gemm_template_cu_template.format(
                    M=case[0],
                    K=case[1],
                    N=n,
                    EXPERTS=case[2],
                    TYPE=type,
                    PADDING=case[3],
                    GROUPSIZE=case[4],
                    cutlass_type=get_cutlass_type(type),
                )
            )

            template_cu_file.close()

for type in dtype:
    template_head_file.write("\n")
    template_head_file.write(
        """#define GEMM_SWITCH_{TYPE}(_M, _K, _EXPERTS, _TokenPaddingSize, _kBlockN, _GROUPSIZE, ...)  {{        \\
    if (_M == 0 && _K == 0 && _EXPERTS == 0 && _TokenPaddingSize == 0 && _kBlockN == 0 && _GROUPSIZE == 0) {{    \\""".format(
            TYPE=type
        )
    )

    template_head_file.write("\n")

    for case in gemm_case:
        for n in n_range:
            template_head_file.write(
                """    }} else if (_M == {M} && _K == {K} && _EXPERTS == {EXPERTS} && _TokenPaddingSize == {PADDING} && _kBlockN == {N} && _GROUPSIZE == {GROUPSIZE}) {{                        \\
        w4afp8_gemm_M{M}_N{N}_G{GROUPSIZE}_K{K}_E{EXPERTS}_P{PADDING}_{TYPE}(__VA_ARGS__);  \\""".format(
                    M=case[0], K=case[1], N=n, EXPERTS=case[2], TYPE=type, PADDING=case[3], GROUPSIZE=case[4]
                )
            )
            template_head_file.write("\n")

    template_head_file.write(
        """    } else {   \\
            PADDLE_THROW(phi::errors::Unimplemented("W4aFp8 not supported m=%d k=%d  experts=%d  token_padding_size=%d  kBlockN=%d  groupsize=%d, please add [%d, %d, %d, %d, %d] to the gemm_case array in the custom_ops/utils/auto_gen_w4afp8_gemm_kernel.py file and recompile it\\n", _M, _K, _EXPERTS, _TokenPaddingSize, _kBlockN, _GROUPSIZE, _M, _K, _EXPERTS, _TokenPaddingSize, _GROUPSIZE));    \\
        }     \\
    }"""
    )

template_head_file.close()
