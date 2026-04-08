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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include "helper.h"

__device__ __forceinline__ void hadamard32_warp(__nv_bfloat16& x) {
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int lane_mask = 1 << step;
        const __nv_bfloat16 sign = (lane_id & lane_mask) ? -1.f : 1.f;
        __nv_bfloat16 x_val_other = __shfl_xor_sync(0xffffffff, x, lane_mask);
        x = sign * x + x_val_other;
    }
}

__global__ void MoeFusedHadamardQuantFp8Kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scale,
    const int64_t* __restrict__ topk_ids,
    __nv_fp8_e4m3* out,
    const int top_k,
    const int intermediate_size,
    const int64_t numel
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= numel) return;

    int64_t token_idx = out_idx / (top_k * intermediate_size);
    int64_t topk_idx  = (out_idx / intermediate_size) % top_k;
    int64_t inter_idx = out_idx % intermediate_size;

    int64_t input_idx = token_idx * intermediate_size + inter_idx;
    if (input_idx >= numel / top_k) return;

    int64_t expert_id = topk_ids[token_idx * top_k + topk_idx];
    float scale_value = scale[expert_id];

    __nv_bfloat16 x = input[input_idx];
    hadamard32_warp(x);

    float x_fp32 = __bfloat162float(x);
    float quantized = x_fp32 / scale_value;
    out[out_idx] = static_cast<__nv_fp8_e4m3>(quantized);
}

__global__ void MoeFusedHadamardQuantFp8TiledKernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scale,
    const int64_t* __restrict__ topk_ids,
    __nv_fp8_e4m3* out,
    const int top_k,
    const int intermediate_size,
    const int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int64_t token_idx = idx / intermediate_size;
    int64_t expert_id = topk_ids[token_idx];
    float scale_value = scale[expert_id];

    __nv_bfloat16 x = input[idx];
    hadamard32_warp(x);

    float x_fp32 = __bfloat162float(x);
    float quantized = x_fp32 / scale_value;
    out[idx] = static_cast<__nv_fp8_e4m3>(quantized);
}

std::vector<paddle::Tensor> MoeFusedHadamardQuantFp8(
                                const paddle::Tensor &input,
                                const paddle::Tensor &scale,
                                const paddle::Tensor &topk_ids,
                                const int top_k,
                                const int intermediate_size,
                                const bool tiled) {
    int64_t numel = input.numel();
    if (!tiled) numel *= top_k;
    paddle::Tensor out = GetEmptyTensor(
            {numel / intermediate_size, intermediate_size},
            paddle::DataType::FLOAT8_E4M3FN,
            input.place());
    constexpr int64_t thread_per_block = 256;
    int64_t block_per_grid = (numel + thread_per_block - 1) / thread_per_block;
    auto stream = input.stream();
    if (tiled) {
        MoeFusedHadamardQuantFp8TiledKernel<<<block_per_grid, thread_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<paddle::bfloat16>()),
            scale.data<float>(),
            topk_ids.data<int64_t>(),
            reinterpret_cast<__nv_fp8_e4m3*>(out.mutable_data<phi::dtype::float8_e4m3fn>()),
            top_k,
            intermediate_size,
            numel
        );
    } else {
        MoeFusedHadamardQuantFp8Kernel<<<block_per_grid, thread_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<phi::dtype::bfloat16>()),
            scale.data<float>(),
            topk_ids.data<int64_t>(),
            reinterpret_cast<__nv_fp8_e4m3*>(out.mutable_data<phi::dtype::float8_e4m3fn>()),
            top_k,
            intermediate_size,
            numel
        );
    }
    return {out};
}

PD_BUILD_STATIC_OP(moe_fused_hadamard_quant_fp8)
    .Inputs({"input", "scale", "topk_ids"})
    .Outputs({"output"})
    .Attrs({"top_k: int",
            "intermediate_size: int",
            "tiled: bool"})
    .SetKernelFn(PD_KERNEL(MoeFusedHadamardQuantFp8));


paddle::Tensor MoeFusedHadamardQuantFp8Func(
                const paddle::Tensor &input,
                const paddle::Tensor &scale,
                const paddle::Tensor &topk_ids,
                const int top_k,
                const int intermediate_size,
                const bool tiled) {
    return MoeFusedHadamardQuantFp8(input, scale, topk_ids, top_k, intermediate_size, tiled)[0];
}


__global__ void FusedHadamardQuantFp8Kernel(
                              const __nv_bfloat16* __restrict__ input,
                              __nv_fp8_e4m3* out,
                              const float scale,
                              const int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    __nv_bfloat16 x = input[idx];
    hadamard32_warp(x);

    float x_fp32 = __bfloat162float(x);
    float quantized = x_fp32 / scale;
    out[idx] = static_cast<__nv_fp8_e4m3>(quantized);
}

std::vector<paddle::Tensor> FusedHadamardQuantFp8(
                                const paddle::Tensor &input,
                                const float scale) {
    int64_t numel = input.numel();
    paddle::Tensor out = GetEmptyTensor(
            input.dims(),
            paddle::DataType::FLOAT8_E4M3FN,
            input.place());
    constexpr int64_t thread_per_block = 256;
    int64_t block_per_grid = (numel + thread_per_block - 1) / thread_per_block;
    auto stream = input.stream();
    FusedHadamardQuantFp8Kernel<<<block_per_grid, thread_per_block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data<paddle::bfloat16>()),
        reinterpret_cast<__nv_fp8_e4m3*>(out.mutable_data<phi::dtype::float8_e4m3fn>()),
        scale,
        numel
    );
    return {out};
}

PD_BUILD_STATIC_OP(fused_hadamard_quant_fp8)
    .Inputs({"input"})
    .Outputs({"output"})
    .Attrs({"scale: float"})
    .SetKernelFn(PD_KERNEL(FusedHadamardQuantFp8));


paddle::Tensor FusedHadamardQuantFp8Func(
                const paddle::Tensor &input,
                const float scale) {
    return FusedHadamardQuantFp8(input, scale)[0];
}
