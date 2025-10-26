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

#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "kernel_traits.h"
#include "mainloop_fwd.h"

template <typename Ktraits>
void __global__ __launch_bounds__(Ktraits::kNWarps *cutlass::NumThreadsPerWarp,
                                  1)
    w4afp8_gemm_kernel(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloopFwd<Ktraits>::Params const mainloop_params) {
  using Element = typename Ktraits::Element;
  static_assert(cutlass::sizeof_bits_v<Element> == 8);

  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int M = Ktraits::M;
  static constexpr int K = Ktraits::K;
  static constexpr int TokenPackSize = Ktraits::TokenPackSize;
  static constexpr int WeightScaleGroup = Ktraits::WeightScaleGroup;

  using CollectiveMainloop = CollectiveMainloopFwd<Ktraits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using ElementOutput = typename Ktraits::ElementOutput;

  extern __shared__ char shared_memory[];
  auto &shared_storage =
      *reinterpret_cast<typename Ktraits::SharedStorage *>(shared_memory);

  const int bidm = blockIdx.x;
  const int bidn = blockIdx.y;
  const int bidb = blockIdx.z;
  const int tidx = threadIdx.x;

  if (tidx == 0) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  if constexpr (WeightScaleGroup == K) {
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesA +
        CollectiveMainloop::TmaTransactionBytesB;
  } else {
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesA +
        CollectiveMainloop::TmaTransactionBytesB +
        CollectiveMainloop::TmaTransactionBytesScale;
  }
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(
      shared_storage.pipeline, pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;

  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  const int pre_fix_tokens =
      TokenPackSize == 0 ? (bidb == 0 ? 0 : mainloop_params.tokens[bidb - 1])
                         : 0;

  const int tokens = TokenPackSize == 0
                         ? mainloop_params.tokens[bidb] - pre_fix_tokens
                         : mainloop_params.tokens[bidb];

  if (bidn * kBlockN >= tokens) {
    return;
  }

  const bool is_need_input_scale = mainloop_params.input_scale != nullptr;

  float *input_scale =
      is_need_input_scale
          ? reinterpret_cast<float *>(shared_memory +
                                      sizeof(typename Ktraits::SharedStorage))
          : nullptr;

  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 40 : 32>();
    PipelineState smem_pipe_write =
        cutlass::make_producer_start_state<MainloopPipeline>();
    collective_mainloop.load(mainloop_params,
                             pipeline,
                             smem_pipe_write,
                             shared_storage,
                             tokens,
                             pre_fix_tokens,
                             bidm,
                             bidn,
                             bidb,
                             tidx);
  } else {
    cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 232 : 160>();
    PipelineState smem_pipe_read;

    typename Ktraits::TiledMma tiled_mma;

    const int mma_tidx = tidx - NumCopyThreads;

    if (is_need_input_scale) {
      if constexpr (TokenPackSize == 0) {
        const int input_scale_idx = pre_fix_tokens + bidn * kBlockN;
        if (mma_tidx < tokens) {
          reinterpret_cast<float *>(input_scale)[mma_tidx] =
              reinterpret_cast<const float *>(mainloop_params.input_scale +
                                              input_scale_idx)[mma_tidx];
        }
      } else {
        const int input_scale_idx = bidb * TokenPackSize + bidn * kBlockN;
        if (mma_tidx < kBlockN / 4) {
          reinterpret_cast<float4 *>(input_scale)[mma_tidx] =
              reinterpret_cast<const float4 *>(mainloop_params.input_scale +
                                               input_scale_idx)[mma_tidx];
        }
      }
    }

    float2 weight_scale;

    if constexpr (WeightScaleGroup == K) {
      weight_scale = reinterpret_cast<const float2 *>(
          mainloop_params.weight_scale + bidb * M +
          bidm * kBlockM)[mma_tidx / 4];
    }
    Tensor tSrS =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));

    if constexpr (WeightScaleGroup == K) {
      collective_mainloop.mma(mainloop_params,
                              tiled_mma,
                              pipeline,
                              smem_pipe_read,
                              shared_storage,
                              tSrS,
                              mma_tidx);
    } else {
      collective_mainloop.mma_pipeline(mainloop_params,
                                       tiled_mma,
                                       pipeline,
                                       smem_pipe_read,
                                       shared_storage,
                                       tSrS,
                                       mma_tidx);
    }

    collective_mainloop.store(mainloop_params,
                              tSrS,
                              shared_storage,
                              tiled_mma,
                              reinterpret_cast<const float *>(&weight_scale),
                              input_scale,
                              tokens,
                              pre_fix_tokens,
                              bidm,
                              bidn,
                              bidb,
                              mma_tidx);
  }
}

template <int Experts>
auto get_gmem_layout(const int Rows, const int Cols) {
  return make_layout(make_shape(static_cast<int64_t>(Rows),
                                static_cast<int64_t>(Cols),
                                static_cast<int64_t>(Experts)),
                     make_stride(static_cast<int64_t>(Cols),
                                 cute::_1{},
                                 static_cast<int64_t>(Rows * Cols)));
}

template <int Experts>
auto get_scale_layout(const int Rows, const int Cols) {
  return make_layout(make_shape(static_cast<int64_t>(Cols),
                                static_cast<int64_t>(Rows),
                                static_cast<int64_t>(Experts)),
                     make_stride(cute::_1{},
                                 static_cast<int64_t>(Cols),
                                 static_cast<int64_t>(Rows * Cols)));
}

template <typename InputType,
          typename OutputType,
          typename Kernel_traits,
          int M,
          int K,
          int Experts,
          int TokenPackSize,
          int WeightScaleGroup>
void run_gemm(const InputType *A,
              const InputType *B,
              OutputType *C,
              const float *weight_scale,
              const float *input_dequant_scale,
              const int64_t *tokens,
              const int max_tokens,
              cudaStream_t stream) {
  using ElementOutput = typename Kernel_traits::ElementOutput;
  using Element = typename Kernel_traits::Element;
  using CollectiveMainloop = CollectiveMainloopFwd<Kernel_traits>;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

  constexpr int M_nums =
      (M + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  const int N_nums =
      (max_tokens + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
  constexpr int K_scale_nums = K / Kernel_traits::kBlockM;
  static_assert(K % WeightScaleGroup == 0);
  static_assert(WeightScaleGroup == 128 || WeightScaleGroup == K);

  typename CollectiveMainloop::Params mainloop_params =
      CollectiveMainloop::to_underlying_arguments(
          {static_cast<Element const *>(A),
           get_gmem_layout<Experts>(M, K / 2),
           static_cast<Element const *>(B),
           get_gmem_layout<Experts>(
               TokenPackSize == 0 ? max_tokens : TokenPackSize, K),
           static_cast<ElementOutput *>(C),
           get_gmem_layout<Experts>(
               M, TokenPackSize == 0 ? max_tokens : TokenPackSize),
           weight_scale,
           get_scale_layout<Experts>(M_nums,
                                     K_scale_nums * Kernel_traits::kBlockM),
           input_dequant_scale,
           tokens});

  void *kernel;
  kernel = (void *)w4afp8_gemm_kernel<Kernel_traits>;

  int smem_size = sizeof(typename Kernel_traits::SharedStorage) +
                  Kernel_traits::kBlockN * sizeof(float);

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  dim3 grid_dims;
  grid_dims.x = M_nums;
  grid_dims.y = N_nums;
  grid_dims.z = Experts;
  static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}),
                    size<1>(ClusterShape{}),
                    size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{
      grid_dims, block_dims, cluster_dims, smem_size, stream};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params);
}
