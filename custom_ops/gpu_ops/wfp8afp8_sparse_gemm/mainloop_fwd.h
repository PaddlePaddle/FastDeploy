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

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "utils.hpp"

using namespace cute;
template <typename Ktraits>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using ElementOutput = typename Ktraits::ElementOutput;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;
    using ElementAccum = typename Ktraits::ElementAccum;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kBlockK = Ktraits::kBlockK;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kTiles = Ktraits::kTiles;
    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma{});
    static constexpr int TokenPackSize = Ktraits::TokenPackSize;
    static constexpr int M = Ktraits::M;


    using GmemTiledCopy = cute::SM90_TMA_LOAD;
    using GmemTiledCopyStore = cute::SM90_TMA_STORE;

    using SmemLayoutA = typename Ktraits::SmemLayoutA;
    using SmemLayoutB = typename Ktraits::SmemLayoutB;
    using SmemLayoutC = typename Ktraits::SmemLayoutC;
    using SmemLayoutE = typename Ktraits::SmemLayoutE;
    using SmemLayoutB_TAIL = typename Ktraits::SmemLayoutB_TAIL;

    using ShapeT = cute::Shape<int64_t, int64_t, int64_t>;
    using StrideT = cute::Shape<int64_t, _1, int64_t>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using WShapeT = cute::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using WStrideT = cute::Shape<int64_t, _1, int64_t, int64_t, int64_t>;
    using WLayoutT = cute::Layout<WShapeT, WStrideT>;

    using EShapeT = cute::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using EStrideT = cute::Shape<_1, int64_t, int64_t, int64_t, int64_t>;
    using ELayoutT = cute::Layout<EShapeT, EStrideT>;

    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)),
            WShapeT{},
            WStrideT{}
        ),
        SmemLayoutA{}(_, _, _0{}),
        select<0, 1>(Shape<Int<kBlockM / 2>, Int<kBlockK>>{}),
        size<0>(ClusterShape{})));

    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)),
            ShapeT{},
            StrideT{}
        ),
        take<0, 2>(SmemLayoutB{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{})));

    using TMA_E = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(
            make_gmem_ptr(static_cast<uint32_t const*>(nullptr)),
            EShapeT{},
            EStrideT{}
        ),
        SmemLayoutE{}(_, _, _0{}),
        select<0, 1>(Shape<Int<NumMmaThreads>, Int<kBlockK / 64>>{}),
        size<0>(ClusterShape{})));

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesE = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutE{})) * cutlass::sizeof_bits_v<int> / 8);

    struct Arguments {
        Element const* ptr_A;
        WLayoutT layout_A;
        uint32_t const* ptr_E;
        ELayoutT layout_E;
        Element const* ptr_B;
        LayoutT layout_B;
        ElementOutput * ptr_C;
        LayoutT layout_C;
        const int *tokens;
        const float *weight_scale;
    };

    struct Params {
        WLayoutT layout_A;
        ELayoutT layout_E;
        LayoutT layout_B;
        TMA_A tma_load_A;
        TMA_E tma_load_E;
        TMA_B tma_load_B;
        const int *tokens;
        const float *weight_scale;
        ElementOutput * ptr_C;
    };


    Params static
    to_underlying_arguments(Arguments const& args) {
        Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), args.layout_A);
        TMA_A tma_load_A = make_tma_copy(
            GmemTiledCopy{},
            mA,
            SmemLayoutA{}(_, _, _0{}),
            select<0, 1>(Shape<Int<kBlockM / 2>, Int<kBlockK>>{}),
            size<0>(ClusterShape{}));
        Tensor mE = make_tensor(make_gmem_ptr(args.ptr_E), args.layout_E);
        TMA_E tma_load_E = make_tma_copy(
            GmemTiledCopy{},
            mE,
            SmemLayoutE{}(_, _, _0{}),
            select<0, 1>(Shape<Int<NumMmaThreads>, Int<kBlockK / 64>>{}),
            size<0>(ClusterShape{}));
        Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), args.layout_B);
        TMA_B tma_load_B = make_tma_copy(
            GmemTiledCopy{},
            mB,
            SmemLayoutB{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{}));

        return {args.layout_A, args.layout_E, args.layout_B,
            tma_load_A, tma_load_E, tma_load_B,
            args.tokens, args.weight_scale, args.ptr_C};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_A.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_B.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_E.get_tma_descriptor());
    }

    template <int CUR_N, typename SharedStorage>
    CUTLASS_DEVICE void
    store(Params const& mainloop_params,
        float * acc_s,
        SharedStorage& shared_storage,
        const int pre_fix_tokens,
        const int tokens,
        const float * weight_scale,
        const int bidm,
        const int bidn,
        const int bidb,
        const int tidx) {
        typename Ktraits::TiledMma tiled_mma;
        using packHalf = typename PackedHalf<ElementOutput>::Type;
        Tensor tOrO_out = make_tensor<ElementOutput>(partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{})).layout());

        #pragma unroll
        for (int i = 0; i < size(tOrO_out); i+=4) {
            acc_s[i] *= weight_scale[0];
            acc_s[i + 1] *= weight_scale[0];
            acc_s[i + 2] *= weight_scale[1];
            acc_s[i + 3] *= weight_scale[1];
            *reinterpret_cast<packHalf*>(&tOrO_out[i]) = packHalf(acc_s[i], acc_s[i + 2]);
            *reinterpret_cast<packHalf*>(&tOrO_out[i + 2]) = packHalf(acc_s[i + 1], acc_s[i + 3]);
        }

        uint16_t *smem_c = reinterpret_cast<uint16_t *>(shared_storage.smem_c.data());

        uint32_t * reg_data = reinterpret_cast<uint32_t*>(tOrO_out.data());

        cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0);

        constexpr int k_copy_times = CUR_N / 16;

        #pragma unroll
        for (int i = 0; i < k_copy_times; i++) {
            uint32_t smem_ptr = cast_smem_ptr_to_uint(reinterpret_cast<uint128_t*>(smem_c + i * 16 * 128) + tidx);
            #if defined(CUTE_ARCH_STSM_SM90_ENABLED)
            asm volatile (
                "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(smem_ptr), "r"(reg_data[4 * i + 0]), "r"(reg_data[4 * i + 2]), "r"(reg_data[4 * i + 1]), "r"(reg_data[4 * i + 3]));
            #endif
        }

        cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0);
        const int batch_idx = TokenPackSize == 0 ? pre_fix_tokens * M : bidb * M * TokenPackSize;
        ElementOutput * store_c = mainloop_params.ptr_C + batch_idx + bidn * (M * kBlockN) + bidm * kBlockM;

        const int reamin_tokens = tokens - bidn * kBlockN;

        const int col = tidx % 2;

        constexpr int kPackSize = 16 / sizeof(ElementOutput);
        constexpr int kNumVecElem = kBlockM / kPackSize;
        constexpr int copy_len = CUR_N * kNumVecElem;
        #pragma unroll
        for (int idx = tidx; idx < copy_len; idx += NumMmaThreads) {
            const int idx_div2 = idx / 2;
            const int store_idx = idx_div2 / 128 * 128 + idx_div2 % 8 * 16 + idx_div2 % 128 / 16 + idx_div2 % 16 / 8 * 8;
            const int store_global_idx = store_idx * 2 + col;
            const int row = store_global_idx / kNumVecElem;
            const int col = store_global_idx % kNumVecElem;
            if (row >= reamin_tokens) {
                continue;
            }
            const int offset = row * (M / kPackSize) + col;
            reinterpret_cast<uint4*>(store_c)[offset] = reinterpret_cast<uint4*>(smem_c)[idx];
        }
    }

    template <typename MTensor>
    CUTLASS_DEVICE auto get_local_packed_tensor(
        const MTensor &mB,
        const int tokens,
        const int bidn) const {

        auto mB_this_batch = make_tensor(
            mB.data(),
            make_layout(
                cute::make_shape(tokens, size<1>(mB)),
                mB.stride()
            ));
        return local_tile(mB_this_batch, select<1, 2>(TileShape_MNK{}), make_coord(bidn, _));
    }

    template <typename MTensor>
    CUTLASS_DEVICE auto get_local_no_packed_tensor(
        const MTensor &mB,
        const int pre_fix_token,
        const int actual_token,
        const int bidn) const {

        auto g_offset = local_tile(
            mB(_, _, 0),
            cute::make_shape(1, size<1>(mB)),
            make_coord(pre_fix_token, _0{}));

        auto g_tensor = make_tensor(
            g_offset.data(),
            make_layout(
                cute::make_shape(actual_token, size<1>(mB)),
                g_offset.stride()
            ));

        Tensor gB = local_tile(g_tensor, select<1, 2>(TileShape_MNK{}), make_coord(bidn, _));

        return gB;
    }


    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         const int pre_fix_tokens,
         const int tokens,
         const int bidm,
         const int bidn,
         const int bidb,
         const int tidx) {

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), SmemLayoutB{});
        Tensor sE = make_tensor(make_smem_ptr(shared_storage.smem_e.data()), SmemLayoutE{});

        Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(mainloop_params.layout_A.shape());
        Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(mainloop_params.layout_B.shape());
        Tensor mE = mainloop_params.tma_load_E.get_tma_tensor(mainloop_params.layout_E.shape());

        Tensor gA = local_tile(mA(_, _, _, bidm, bidb), select<0, 1>(Shape<Int<kBlockM / 2>, Int<kBlockK>>{}), make_coord(0,0,_));

        Tensor gE = local_tile(mE(_, _, _, bidm, bidb), select<0, 1>(Shape<Int<NumMmaThreads>, Int<kBlockK / 64>>{}), make_coord(0, 0));

        auto [tAgA, tAsA] = tma_partition(mainloop_params.tma_load_A, _0{}, Layout<ClusterShape>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));

        auto [tEgE, tEsE] = tma_partition(mainloop_params.tma_load_E, _0{}, Layout<ClusterShape>{},  group_modes<0, 2>(sE), group_modes<0, 2>(gE));

        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if constexpr (TokenPackSize == 0) {
            Tensor gB = get_local_no_packed_tensor(
                mB,
                pre_fix_tokens,
                tokens,
                bidn);
            auto [tBgB, tBsB] = tma_partition(mainloop_params.tma_load_B, _0{}, Layout<ClusterShape>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));

            const int kIters = kTiles / kStages;
            if (tidx == 0) {
                #pragma unroll
                for (int kiter = 0; kiter < kIters; ++kiter) {
                    #pragma unroll
                    for (int s = 0; s < kStages; s++) {
                        const int i = kiter * kStages + s;
                        pipeline.producer_acquire(smem_pipe_write);
                        copy(mainloop_params.tma_load_A.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tAgA(_, i), tAsA(_, s));
                        copy(mainloop_params.tma_load_E.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tEgE(_, i), tEsE(_, s));
                        copy(mainloop_params.tma_load_B.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tBgB(_, i), tBsB(_, s));
                        ++smem_pipe_write;
                    }
                }

                #pragma unroll
                for (int i = kIters * kStages; i < kTiles; ++i) {
                    pipeline.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_A.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tAgA(_, i), tAsA(_, smem_pipe_write.index()));
                    copy(mainloop_params.tma_load_E.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tEgE(_, i), tEsE(_, smem_pipe_write.index()));
                    copy(mainloop_params.tma_load_B.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tBgB(_, i), tBsB(_, smem_pipe_write.index()));
                    ++smem_pipe_write;
                }
            }
        } else {
            auto mB_this_batch = make_tensor(
                mB(_, _, bidb).data(),
                make_layout(
                    cute::make_shape(tokens, size<1>(mB)),
                    mB.stride()
                ));
            Tensor gB = local_tile(mB_this_batch, select<1, 2>(TileShape_MNK{}), make_coord(bidn, _));
            auto [tBgB, tBsB] = tma_partition(mainloop_params.tma_load_B, _0{}, Layout<ClusterShape>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));

            const int kIters = kTiles / kStages;
            if (tidx == 0) {
                #pragma unroll
                for (int kiter = 0; kiter < kIters; ++kiter) {
                    #pragma unroll
                    for (int s = 0; s < kStages; s++) {
                        const int i = kiter * kStages + s;
                        pipeline.producer_acquire(smem_pipe_write);
                        copy(mainloop_params.tma_load_A.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tAgA(_, i), tAsA(_, s));
                        copy(mainloop_params.tma_load_E.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tEgE(_, i), tEsE(_, s));
                        copy(mainloop_params.tma_load_B.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                        tBgB(_, i), tBsB(_, s));
                        ++smem_pipe_write;
                    }
                }

                #pragma unroll
                for (int i = kIters * kStages; i < kTiles; ++i) {
                    pipeline.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_A.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tAgA(_, i), tAsA(_, smem_pipe_write.index()));
                    copy(mainloop_params.tma_load_E.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tEgE(_, i), tEsE(_, smem_pipe_write.index()));
                    copy(mainloop_params.tma_load_B.with(*pipeline.producer_get_barrier(smem_pipe_write), 0),
                    tBgB(_, i), tBsB(_, smem_pipe_write.index()));
                    ++smem_pipe_write;
                }
            }
        }
    }

    template <int CUR_N, typename SharedStorage>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
            MainloopPipeline pipeline,
            PipelineState& smem_pipe_read,
            SharedStorage& shared_storage,
            float *acc_s,
            const int tidx) {

        using sMemBLayout = std::conditional_t<
            CUR_N == kBlockN,
            SmemLayoutB,
            SmemLayoutB_TAIL
        >;

        using Mma = std::conditional_t<
            CUR_N == kBlockN,
            typename Ktraits::Mma,
            typename Ktraits::Mma_TAIL
        >;

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), sMemBLayout{});
        Tensor sE = make_tensor(make_smem_ptr(shared_storage.smem_e.data()), SmemLayoutE{});

        const int wg_idx = tidx / 128;
        const int wg_offset = wg_idx * 64;

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        constexpr int E_STEP = kBlockK / 64 * NumMmaThreads;
        constexpr int B_STEPS = CUR_N == 0 ? 1 : (kBlockN / CUR_N);

        const int kIters = kTiles / kStages;
        #pragma unroll
        for (int kiter = 0; kiter < kIters; ++kiter) {
            #pragma unroll
            for (int s = 0; s < kStages; s++) {
                consumer_wait(pipeline, smem_pipe_read);

                gemm<Mma, kBlockK, NumMmaThreads>(
                    sA(_, _, s).data().get().get() + wg_offset,
                    sB(_, _, s * B_STEPS).data().get().get(),
                    acc_s,
                    shared_storage.smem_e.data() + s * E_STEP + tidx);

                pipeline.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }
        }

        #pragma unroll
        for (int i = 0; i < kTiles % kStages; ++i) {
            consumer_wait(pipeline, smem_pipe_read);

            gemm<Mma, kBlockK, NumMmaThreads>(
                sA(_, _, i).data().get().get() + wg_offset,
                sB(_, _, i * B_STEPS).data().get().get(),
                acc_s,
                shared_storage.smem_e.data() + i * E_STEP + tidx);

            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }
    }

};
