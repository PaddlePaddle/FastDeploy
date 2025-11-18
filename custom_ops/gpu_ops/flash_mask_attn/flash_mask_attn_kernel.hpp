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

#pragma once
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cluster_launch.hpp"

#include "kernel_traits.h"
#include "mainloop_attn.hpp"
#include "softmax.hpp"

using namespace cute;

template <int kHeadDim>
auto get_gmem_layout(int token_num, int head_num) {
  return make_layout(make_shape(token_num, kHeadDim, head_num),
                     make_stride(head_num * kHeadDim, cute::_1{}, kHeadDim));
}

template <typename Ktraits>
__global__ void __launch_bounds__(Ktraits::kNWarps *cutlass::NumThreadsPerWarp,
                                  1)
    compute_attn_ws(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloopAttn<Ktraits>::Params const mainloop_params,
        CUTE_GRID_CONSTANT Flash_mask_params const data_params) {
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using output_type = typename Ktraits::output_type;
  using SoftType = ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  constexpr int kHeadDim = Ktraits::kHeadDim;
  constexpr bool NeedMask = Ktraits::NeedMask;

  using CollectiveMainloop = CollectiveMainloopAttn<Ktraits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto &shared_storage =
      *reinterpret_cast<typename Ktraits::SharedStorage *>(shared_memory);

  __align__(16) __shared__ int mask[kBlockM];

  const int m_block = blockIdx.x;
  const int bidh = blockIdx.y;
  const int bidb = blockIdx.z;

  if constexpr (NeedMask) {
    const int *mask_this_batch =
        data_params.mask + data_params.cu_seq_q[bidb] + m_block * kBlockM;

    for (int i = threadIdx.x; i < kBlockM;
         i += Ktraits::kNWarps * cutlass::NumThreadsPerWarp) {
      mask[i] = mask_this_batch[i];
    }
  }

  const int seq_len_q = data_params.seq_len_encoder[bidb];
  const int seq_len_k =
      data_params.cu_seq_k[bidb + 1] - data_params.cu_seq_k[bidb];

  if (m_block * kBlockM >= seq_len_q) {
    return;
  }

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumMmaThreads;

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(1);
  }

  MainloopPipeline pipeline_k(
      shared_storage.pipeline_k, pipeline_params, ClusterShape{});
  MainloopPipeline pipeline_v(
      shared_storage.pipeline_v, pipeline_params, ClusterShape{});

  __syncthreads();

  CollectiveMainloop collective_mainloop;

  const int real_seq = seq_len_q - m_block * kBlockM;

  const int n_block_max =
      NeedMask
          ? cute::ceil_div(mask[min(kBlockM - 1, real_seq - 1)], kBlockN)
          : min(cute::ceil_div((m_block + 1) * kBlockM + seq_len_k - seq_len_q,
                               kBlockN),
                cute::ceil_div(seq_len_k, kBlockN));
  ;

  if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 8 ? 56 : 24>();

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k =
          cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v =
          cutlass::make_producer_start_state<MainloopPipeline>();

      collective_mainloop.load(mainloop_params,
                               pipeline_k,
                               pipeline_v,
                               smem_pipe_write_k,
                               smem_pipe_write_v,
                               shared_storage,
                               n_block_max,
                               m_block,
                               bidh,
                               bidb,
                               data_params.cu_seq_q,
                               data_params.cu_seq_k,
                               seq_len_q,
                               seq_len_k);
    }
  } else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 8 ? 256 : 240>();
    typename Ktraits::TiledMma1 tiled_mma1;

    collective_mainloop.mma_init();

    PipelineState smem_pipe_read_k, smem_pipe_read_v;

    Tensor tOrO =
        partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
    Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax;

    collective_mainloop.mma(mainloop_params,
                            pipeline_k,
                            pipeline_v,
                            smem_pipe_read_k,
                            smem_pipe_read_v,
                            tOrO,
                            softmax,
                            mask,
                            n_block_max,
                            threadIdx.x - NumCopyThreads,
                            m_block,
                            seq_len_q,
                            seq_len_k,
                            shared_storage);

    const int o_head_stride = data_params.head_num * kHeadDim;
    const int store_offset =
        (data_params.cu_seq_q[bidb] + m_block * kBlockM) * o_head_stride +
        bidh * kHeadDim;

    collective_mainloop.store<NumMmaThreads>(
        mainloop_params,
        tOrO,
        shared_storage,
        tiled_mma1,
        threadIdx.x - NumCopyThreads,
        o_head_stride,
        real_seq,
        reinterpret_cast<output_type *>(data_params.o_ptr) + store_offset);
  }
}

template <typename Kernel_traits>
void run_flash_mask(Flash_mask_params &params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

  using CollectiveMainloop = CollectiveMainloopAttn<Kernel_traits>;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  typename CollectiveMainloop::Params mainloop_params =
      CollectiveMainloop::to_underlying_arguments(
          {static_cast<Element const *>(params.q_ptr),
           get_gmem_layout<kHeadDim>(params.max_seq_len_q * params.batch_size,
                                     params.head_num),
           static_cast<Element const *>(params.k_ptr),
           get_gmem_layout<kHeadDim>(params.max_seq_len_k * params.batch_size,
                                     params.kv_head_num),
           static_cast<Element const *>(params.v_ptr),
           get_gmem_layout<kHeadDim>(params.max_seq_len_k * params.batch_size,
                                     params.kv_head_num),
           params.scale_softmax_log2});

  int num_blocks_m =
      cutlass::ceil_div(params.max_seq_len_q, Kernel_traits::kBlockM);

  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) *
                 size<0>(ClusterShape{});

  void *kernel;
  kernel = (void *)compute_attn_ws<Kernel_traits>;
  int smem_size = sizeof(typename Kernel_traits::SharedStorage);

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  dim3 grid_dims;
  grid_dims.x = num_blocks_m;
  grid_dims.y = params.head_num;
  grid_dims.z = params.batch_size;

  static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}),
                    size<1>(ClusterShape{}),
                    size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{
      grid_dims, block_dims, cluster_dims, smem_size, stream};
  cutlass::launch_kernel_on_cluster(
      launch_params, kernel, mainloop_params, params);
}

template <int kBlockM,
          int kBlockN,
          bool NeedMask,
          typename InputType,
          typename OutputType>
void flash_attn_headdim128(Flash_mask_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 128;
  constexpr static int kNWarps = kBlockM / 16 + 4;
  constexpr static int kStages = 2;

  using Ktraits = Flash_mask_kernel_traits<Headdim,
                                           kBlockM,
                                           kBlockN,
                                           kNWarps,
                                           kStages,
                                           NeedMask,
                                           InputType,
                                           OutputType>;
  run_flash_mask<Ktraits>(params, stream);
}
