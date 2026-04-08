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
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/reg_reconfig.h"

#include "kernel_traits.h"
#include "mainloop_fwd.h"

template <typename Ktraits>
void  __global__ __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1) w8a8_sparse_gemm_kernel(
        CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits>::Params const mainloop_params) {

    using Element = typename Ktraits::Element;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma{});
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int TokenPackSize = Ktraits::TokenPackSize;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int TAIL_N = Ktraits::TAIL_N;
    static constexpr int M = Ktraits::M;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesA + CollectiveMainloop::TmaTransactionBytesE + CollectiveMainloop::TmaTransactionBytesB;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params, ClusterShape{});

    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesB;

    CollectiveMainloop collective_mainloop;

    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }


    const int bidm = blockIdx.x;
    const int bidn = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    const int pre_fix_tokens = TokenPackSize == 0 ? mainloop_params.tokens[bidb] : 0;

    const int tokens = TokenPackSize == 0 ? mainloop_params.tokens[bidb + 1] - pre_fix_tokens : mainloop_params.tokens[bidb];


    if (bidn * kBlockN >= tokens) {
        return;
    }

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
        collective_mainloop.load(
                mainloop_params,
                pipeline,
                smem_pipe_write,
                shared_storage,
                pre_fix_tokens,
                tokens,
                bidm,
                bidn,
                bidb,
                tidx);
    } else {
        cutlass::arch::warpgroup_reg_alloc<232>();
        PipelineState smem_pipe_read;

        constexpr int acc_num = sizeof(typename Ktraits::Mma::CRegisters) / sizeof(float);
        float acc_s[acc_num];

        #pragma unroll
        for (int i = 0; i < acc_num; ++i) {
            acc_s[i] = 0.0f;
        }

        const int reamin_tokens = tokens - bidn * kBlockN;

        const int mma_tidx = tidx - NumCopyThreads;

        const float2 weight_scale = reinterpret_cast<const float2*>(mainloop_params.weight_scale + bidb * M + bidm * kBlockM)[mma_tidx / 4];


        if (TAIL_N > 0 && reamin_tokens < kBlockN) {
            collective_mainloop.mma<TAIL_N>(
                mainloop_params,
                pipeline,
                smem_pipe_read,
                shared_storage,
                acc_s,
                mma_tidx);

            collective_mainloop.store<TAIL_N>(
                mainloop_params,
                acc_s,
                shared_storage,
                pre_fix_tokens,
                tokens,
                reinterpret_cast<const float*>(&weight_scale),
                bidm,
                bidn,
                bidb,
                mma_tidx);
        } else {
            collective_mainloop.mma<kBlockN>(
                mainloop_params,
                pipeline,
                smem_pipe_read,
                shared_storage,
                acc_s,
                mma_tidx);

            collective_mainloop.store<kBlockN>(
                mainloop_params,
                acc_s,
                shared_storage,
                pre_fix_tokens,
                tokens,
                reinterpret_cast<const float*>(&weight_scale),
                bidm,
                bidn,
                bidb,
                mma_tidx);
        }
    }

}

template <int Batch>
auto get_gmem_layout(int Rows, int Cols) {
    return  make_layout(
                make_shape(
                    static_cast<int64_t>(Rows),
                    static_cast<int64_t>(Cols),
                    static_cast<int64_t>(Batch)),
                make_stride(
                    static_cast<int64_t>(Cols),
                    cute::_1{},
                    static_cast<int64_t>(Rows * Cols)));
}

template <int Batch>
auto get_weight_gmem_layout(int m_nums, int k_nums, int Rows, int Cols) {
    return  make_layout(
                make_shape(
                    static_cast<int64_t>(Rows),
                    static_cast<int64_t>(Cols),
                    static_cast<int64_t>(k_nums),
                    static_cast<int64_t>(m_nums),
                    static_cast<int64_t>(Batch)),
                make_stride(
                    static_cast<int64_t>(Cols),
                    cute::_1{},
                    static_cast<int64_t>(Rows * Cols),
                    static_cast<int64_t>(Rows * Cols * k_nums),
                    static_cast<int64_t>(Rows * Cols * k_nums * m_nums)));
}

template <int Batch>
auto get_gmem_e_layout(int ms, int ks, int ks_in, int Cols) {
    return  make_layout(
                make_shape(
                    static_cast<int64_t>(Cols),
                    static_cast<int64_t>(ks_in),
                    static_cast<int64_t>(ks),
                    static_cast<int64_t>(ms),
                    static_cast<int64_t>(Batch)),
                make_stride(
                    cute::_1{},
                    static_cast<int64_t>(Cols),
                    static_cast<int64_t>(ks_in * Cols),
                    static_cast<int64_t>(ks * ks_in * Cols),
                    static_cast<int64_t>(ms * ks * Cols * 2)));
}

template <typename InputType, typename OutputType, typename Kernel_traits, int M, int K, int Batch, int kPackTokenSize>
void run_gemm(
        const InputType * A,
        const uint32_t *E,
        const InputType * B,
        OutputType * C,
        const float *weight_scale,
        const int *tokens_idx,
        const int max_tokens,
        cudaStream_t stream) {

    using ElementOutput = typename Kernel_traits::ElementOutput;
    using Element = typename Kernel_traits::Element;
    using CollectiveMainloop = CollectiveMainloopFwd<Kernel_traits>;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;
    constexpr int NumMmaThreads = Kernel_traits::NumMmaThreads;
    constexpr int kBlockK = Kernel_traits::kBlockK;
    constexpr int kBlockM = Kernel_traits::kBlockM;

    static_assert(M % Kernel_traits::kBlockM == 0);
    constexpr int M_nums = M / Kernel_traits::kBlockM;
    const int N_nums = (max_tokens + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;

    constexpr int kTiles = Kernel_traits::kTiles;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(A),
            get_weight_gmem_layout<Batch>(M_nums, kTiles, kBlockM / 2, kBlockK),
            static_cast<uint32_t const*>(E),
            get_gmem_e_layout<Batch>(M_nums, kTiles, kBlockK / 64, NumMmaThreads),
            static_cast<Element const*>(B),
            get_gmem_layout<Batch>(kPackTokenSize == 0 ? max_tokens * Batch : kPackTokenSize, K),
            static_cast<ElementOutput*>(C),
            get_gmem_layout<Batch>(M, kPackTokenSize == 0 ? max_tokens : kPackTokenSize),
            tokens_idx,
            weight_scale,
        });

    void *kernel;
    kernel = (void *)w8a8_sparse_gemm_kernel<Kernel_traits>;

    int smem_size = sizeof(typename Kernel_traits::SharedStorage);

    if (smem_size >= 48 * 1024) {
       cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 grid_dims;
    grid_dims.x = M_nums;
    grid_dims.y = N_nums;
    grid_dims.z = Batch;
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(
        launch_params, kernel, mainloop_params);
}

template <typename InputType, typename OutputType, int M, int K, int Batch, int kPackTokenSize>
void w8a8_sparse_gemm(
        const InputType * A,
        const uint32_t * E,
        const InputType * B,
        OutputType * C,
        const float *weight_scale,
        const int *tokens_idx,
        const int max_tokens,
        cudaStream_t stream) {
    constexpr static int kBlockM = 128;
    constexpr static int kBlockK = 128;
    constexpr static int kNWarps = 4 + kBlockM / 16;
    constexpr static int kStages = 5;
    constexpr int kCluster = 1;
    static_assert(K % kBlockK == 0);
    constexpr int kTiles = K / kBlockK;
    const int max_tokens_pack16 = (max_tokens + 31) / 32 * 32;

    using Kernel_traits = Kernel_traits<kBlockM, 256, kBlockK, kNWarps, kStages, kTiles, M, kPackTokenSize, 0, kCluster, InputType, OutputType>;
    run_gemm<InputType, OutputType, Kernel_traits, M, K, Batch, kPackTokenSize>(A, E, B, C, weight_scale, tokens_idx, max_tokens_pack16, stream);
}
