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
#include "paddle/extension.h"
#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "../moba_attn_utils.hpp"

using namespace cute;
template <typename T>
struct moba_decoder_attn_params {
    T *__restrict__ q_input;
    void *__restrict__ cache_k;
    void *__restrict__ cache_v;

    T *__restrict__ attn_out;
    T *__restrict__ partition_attn_out;
    T *__restrict__ cache_k_dequant_scale;
    T *__restrict__ cache_v_dequant_scale;
    T *__restrict__ cache_k_quant_scale;
    T *__restrict__ cache_v_quant_scale;
    T *__restrict__ cache_k_zp;
    T *__restrict__ cache_v_zp;
    int * __restrict__ cu_seq_q;
    float * sums;
    float * maxs;
    int * seq_lens_encoder;
    int * seq_lens_decoder;
    int * block_table;
    int max_input_length;
    int max_seq_len;
    int head_num;
    int kv_head_num;
    int max_num_blocks_per_seq;
    float scale_softmax;
    int batch_size;
    int max_num_partitions;
    float inv_sqrt_dh;
    int *qk_gate_topk_idx_ptr;
    int use_moba_seq_limit;
};

template <typename cute_type_, int DataBits_>
struct CacheKV_quant_traits {
    using cuteType = cute_type_;
    static constexpr int kDataBits = DataBits_;
    static constexpr int kBlockSize = 64;
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockKSmem = 64;
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<
                        Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockSize>, Int<kHeadDim>>{}));

    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = kNWarps * 32;


    static constexpr int kThreadPerValue = 16 / sizeof(cuteType);
    static constexpr int kThreadsPerRow = kHeadDim / kThreadPerValue;

    using GmemLayoutAtom = Layout<
        Shape <Int<kNThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
        Stride<Int<kThreadsPerRow>, _1>>;

    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<
            SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cuteType>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kThreadPerValue>>>{}));

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<cuteType, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;

    using ValLayoutMNK = Layout<Shape<_1,_4,_1>>;

    using PermutationMNK = Tile<_16, Int<16 * kNWarps>, _16>;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        ValLayoutMNK,
        PermutationMNK>;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cuteType>;

    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<kBlockKSmem>, Int<kBlockSize>>,
                           Stride<_1, Int<kBlockKSmem>>>{}));

    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<kHeadDim>, Int<kBlockSize>>{}));

    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, cuteType>;

    static constexpr int kShareMemSize = size(SmemLayoutKV{}) * 2 * sizeof(cuteType);
};

template <int kGqaGroupSize_, int kTileN_, int kMaxN_, typename CacheKV_traits_>
struct moba_decoder_attn_kernel_traits {
    using ElementAccum = float;
    using CacheKV_traits = CacheKV_traits_;
    using cuteType = typename CacheKV_traits::cuteType;
    static constexpr int kDataBits = CacheKV_traits::kDataBits;
    static constexpr int kTileN = kTileN_;
    static constexpr int kMaxN = kMaxN_;
    static constexpr int kGqaGroupSize = kGqaGroupSize_;
    static constexpr int kHeadDim = CacheKV_traits::kHeadDim;
    static constexpr int kHeadDimKV = kHeadDim / (16 / kDataBits);
    static constexpr int kMinGemmM = 16;
    static constexpr int kBlockM = (kGqaGroupSize + kMinGemmM - 1) / kMinGemmM * kMinGemmM;
    static constexpr int kBlockSize = CacheKV_traits::kBlockSize;
    static_assert(kGqaGroupSize <= 16);
    static constexpr int32_t kNWarps = CacheKV_traits::kNWarps;

    static constexpr int kBlockKSmem = CacheKV_traits::kBlockKSmem;
    static constexpr int kBlockKVSmem = kHeadDimKV <= 64 ? kHeadDimKV : 64;
    static_assert(kHeadDim % kBlockKSmem == 0);
    static constexpr int kNReduceWarps = 4;
    static constexpr int kNReduceThreads = kNReduceWarps * 32;


    using SmemLayoutAtomQ = typename CacheKV_traits::SmemLayoutAtomQ;

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutQK = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kBlockSize>>{}));

    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<
                        Shape<Int<8>, Int<kBlockKVSmem>>,
                        Stride<Int<kBlockKVSmem>, _1>>{}));

    using SmemLayoutKV_ = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        Shape<Int<kBlockSize>, Int<kHeadDimKV>>{}));

    using SmemLayoutKV = std::conditional_t<
        kDataBits == 16,
        SmemLayoutKV_,
        decltype(get_nonswizzle_portion(SmemLayoutKV_{}))
    >;

    constexpr static int kBlockKVSize = kDataBits == 4 ? 32 : kBlockSize;
    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<kBlockKSmem>, Int<kBlockKVSize>>,
                           Stride<_1, Int<kBlockKSmem>>>{}));

    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<kHeadDim>, Int<kBlockKVSize>>{}));

    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    static constexpr int kThreadsPerRow = CacheKV_traits::kThreadsPerRow;
    static constexpr int kThreadsKVPerRow = kThreadsPerRow / (16 / kDataBits);
    static constexpr int kNThreads = CacheKV_traits::kNThreads;

    using GmemKVLayoutAtom = Layout<
        Shape<Int<kNThreads / kThreadsKVPerRow>, Int<kThreadsKVPerRow>>,
        Stride<Int<kThreadsKVPerRow>, _1>>;

    using SmemCopyAtom = typename CacheKV_traits::SmemCopyAtom;
    using TiledMma = typename CacheKV_traits::TiledMma;

    static constexpr int kThreadPerValue = CacheKV_traits::kThreadPerValue;

    using GmemTiledCopyQ = typename CacheKV_traits::GmemTiledCopyQ;
    using GmemLayoutAtom = typename CacheKV_traits::GmemLayoutAtom;
    using GmemTiledCopyKV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cuteType>{},
                        GmemKVLayoutAtom{},
                        Layout<Shape<_1, Int<kThreadPerValue>>>{}));


    using SmemCopyAtomTransposed = typename CacheKV_traits::SmemCopyAtomTransposed;

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, cuteType>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kThreadPerValue>>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, cuteType>;

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<
                        Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    static constexpr int kShareMemSize = (size(SmemLayoutQ{}) + size(SmemLayoutQK{}) + size(SmemLayoutKV{}) * 2) * sizeof(cuteType);
};
