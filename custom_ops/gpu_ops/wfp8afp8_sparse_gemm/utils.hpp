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

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>


using namespace cute;


template<typename T>
struct PackedHalf;

template<>
struct PackedHalf<cutlass::half_t> {
    using Type = __half2;
};

template<>
struct PackedHalf<cutlass::bfloat16_t> {
    using Type = nv_bfloat162;
};

template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(
        PointerType smem_ptr,
        int layout_type,
        int leading_byte_offset = 0,
        int stride_byte_offset = 1024) {

    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <typename Mma, size_t ...Idx>
__forceinline__ __device__ static void gemm(uint64_t const& desc_a, uint64_t const& desc_b, float* d, const uint32_t e, std::index_sequence<Idx...>) {
    Mma::fma(desc_a, desc_b, d[Idx]..., e, GMMA::ScaleOut::One);
}

template <typename Mma, int kBlockK, int NumMmaThreads, typename T>
__forceinline__ __device__ void gemm(
        const T * sA,
        const T * sB,
        float * acc_c,
        const uint32_t *E) {

    constexpr int acc_num = sizeof(Mma::CRegisters) / sizeof(float);

    warpgroup_arrive();
    // 选择的下标   对应的16进制
    //    01          4
    //    02          8
    //    03          12
    //    12          9
    //    13          13
    //    23          14
    #pragma unroll
    for (int i = 0; i < kBlockK / 64; i++) {
        GmmaDescriptor a_desc = make_smem_desc(sA + i * 32, 1, 0, 1024);
        GmmaDescriptor b_desc = make_smem_desc(sB + i * 64, 1, 0, 1024);
        gemm<Mma>(a_desc, b_desc, acc_c, E[i * NumMmaThreads], std::make_index_sequence<acc_num>{});
    }

    warpgroup_commit_batch();
    warpgroup_wait<0>();
}
