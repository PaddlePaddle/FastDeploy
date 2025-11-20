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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "utils.hpp"

using namespace cute;

enum class AttnNamedBarriers {
  QueryEmpty = 0,
  ValueEmpty = 1,
  TileCountSmemEmpty = 2,
  TileCountSmemFull = 3,
  WarpSchedulerWG1 = 4,
  WarpSchedulerWG2 = 5,
  WarpSchedulerWG3 = 6,
};

template <typename Ktraits>
struct CollectiveMainloopAttn {
  using Element = typename Ktraits::Element;
  using output_type = typename Ktraits::output_type;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kHeadDim = Ktraits::kHeadDim;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr bool NeedMask = Ktraits::NeedMask;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1, int32_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::
                   sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  using GmemTiledCopyO = typename Ktraits::GmemTiledCopyO;

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtomK{},
                             make_shape(shape<1>(TileShape_MNK{}),
                                        shape<2>(TileShape_MNK{}),
                                        Int<kStages>{})));
  using SmemLayoutV = SmemLayoutK;
  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(cute::composition(
      SmemLayoutV{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
          make_stride(get<1>(TileShape_MNK{}),
                      _1{},
                      Int<size(SmemLayoutV{}(_, _, _0{}))>{}))));
  using SmemLayoutO = typename Ktraits::SmemLayoutO;
  using SmemCopyAtomO = typename Ktraits::SmemCopyAtomO;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)),
                  StrideT{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      _1{}));  // no mcast for Q

  using TMA_KV = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)),
                  StrideT{}),
      take<0, 2>(SmemLayoutK{}),
      select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple
  // issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

  static constexpr bool UseSchedulerBarrier = kHeadDim <= 128;

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    LayoutT layout_Q;
    Element const* ptr_K;
    LayoutT layout_K;
    Element const* ptr_V;
    LayoutT layout_V;
    float const softmax_scale_log2;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_Q tma_load_Q;
    TMA_KV tma_load_K, tma_load_V;
    float const softmax_scale_log2;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopyQ{},
                                     mQ,
                                     SmemLayoutQ{},
                                     select<0, 2>(TileShape_MNK{}),
                                     _1{});
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K);
    TMA_KV tma_load_K = make_tma_copy(
        GmemTiledCopyKV{},
        mK,
        SmemLayoutK{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
    TMA_KV tma_load_V = make_tma_copy(
        GmemTiledCopyKV{},
        mV,
        SmemLayoutV{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    return {args.layout_Q,
            args.layout_K,
            args.layout_V,
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()),
                                               get<2>(args.layout_K.shape()))),
            tma_load_Q,
            tma_load_K,
            tma_load_V,
            args.softmax_scale_log2};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_V.get_tma_descriptor());
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor,
                                            const Shape& tile_shape,
                                            const int* cu_seq_len,
                                            const int bidh,
                                            const int bidb,
                                            const int actual_seq_len) const {
    auto g_offset = local_tile(m_tensor(_, _, bidh),
                               cute::make_shape(1, get<1>(tile_shape)),
                               make_coord(cu_seq_len[bidb], _0{}));
    auto g_sequence = make_tensor(
        g_offset.data(),
        make_layout(cute::make_shape(actual_seq_len, get<1>(tile_shape)),
                    g_offset.stride()));
    auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params,
                           MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v,
                           PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v,
                           SharedStorage& shared_storage,
                           const int n_block_max,
                           const int m_block,
                           const int bidh,
                           const int bidb,
                           const int* cu_seq_q,
                           const int* cu_seq_k,
                           const int seq_len_q,
                           const int seq_len_k) {
    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
        mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
        mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
        mainloop_params.layout_V.shape());
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

    Tensor gQ = get_local_tile_tensor(
        mQ, select<0, 2>(TileShape_MNK{}), cu_seq_q, bidh, bidb, seq_len_q)(
        _, _, m_block);
    Tensor gK = get_local_tile_tensor(
        mK, select<1, 2>(TileShape_MNK{}), cu_seq_k, bidh_kv, bidb, seq_len_k);
    Tensor gV = get_local_tile_tensor(
        mV, select<1, 2>(TileShape_MNK{}), cu_seq_k, bidh_kv, bidb, seq_len_k);

    Tensor sQ_x =
        make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x =
        make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q,
                                      _0{},
                                      Layout<_1>{},
                                      group_modes<0, 2>(sQ_x),
                                      group_modes<0, 2>(gQ_x));
    auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K,
                                      _0{},
                                      Layout<_1>{},
                                      group_modes<0, 2>(sK),
                                      group_modes<0, 2>(gK));
    auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V,
                                      _0{},
                                      Layout<_1>{},
                                      group_modes<0, 2>(sV),
                                      group_modes<0, 2>(gV));

    uint16_t mcast_mask_kv = 0;

    int n_block = n_block_max - 1;

    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(
               reinterpret_cast<
                   cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                   shared_storage.barrier_Q),
               0 /*mcast_mask*/),
           tQgQ,
           tQsQ);
    }

    if (lane_predicate) {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      copy(mainloop_params.tma_load_K.with(
               *pipeline_k.producer_get_barrier(smem_pipe_write_k),
               mcast_mask_kv),
           tKgK(_, n_block),
           tKsK(_, smem_pipe_write_k.index()));
      ++smem_pipe_write_k;
    }

    if (lane_predicate) {
#pragma unroll 2
      for (; n_block > 0; --n_block) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(
                 *pipeline_k.producer_get_barrier(smem_pipe_write_k),
                 mcast_mask_kv),
             tKgK(_, n_block - 1),
             tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_V.with(
                 *pipeline_v.producer_get_barrier(smem_pipe_write_v),
                 mcast_mask_kv),
             tVgV(_, n_block),
             tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    }
    if (lane_predicate) {
      pipeline_v.producer_acquire(smem_pipe_write_v);
      copy(mainloop_params.tma_load_V.with(
               *pipeline_v.producer_get_barrier(smem_pipe_write_v),
               mcast_mask_kv),
           tVgV(_, n_block),
           tVsV(_, smem_pipe_write_v.index()));
      ++smem_pipe_write_v;
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(
          NumMmaThreads,
          static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 +
              cutlass::canonical_warp_group_idx() /*id*/);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 + 1 /*id*/);
    }
    if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > 2) {
        cutlass::arch::NamedBarrier::arrive(
            NumMmaThreads,
            static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 +
                2 /*id*/);
      }
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 +
              (3 - cutlass::canonical_warp_group_idx()) /*id*/);
    } else {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 +
              (cutlass::canonical_warp_group_idx() <= 2
                   ? cutlass::canonical_warp_group_idx() + 1
                   : cutlass::canonical_warp_group_idx() + 1 - 3) /*id*/);
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(AttnNamedBarriers::WarpSchedulerWG1) - 1 +
              (cutlass::canonical_warp_group_idx() <= 1
                   ? cutlass::canonical_warp_group_idx() + 2
                   : cutlass::canonical_warp_group_idx() + 2 - 3) /*id*/);
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE void mma(Params const& mainloop_params,
                          MainloopPipeline pipeline_k,
                          MainloopPipeline pipeline_v,
                          PipelineState& smem_pipe_read_k,
                          PipelineState& smem_pipe_read_v,
                          FrgTensorO& tOrO,
                          Softmax& softmax,
                          const int* mask,
                          const int n_block_max,
                          const int thread_idx,
                          const int m_block,
                          const int seq_len_q,
                          const int seq_len_k,
                          SharedStorage& shared_storage) {
    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()),
                             SmemLayoutVt{});

    typename Ktraits::TiledMma0 tiled_mma0;
    typename Ktraits::TiledMma1 tiled_mma1;
    auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
    auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

    Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
    Tensor tSrK = threadMma0.partition_fragment_B(sK);
    Tensor tOrV = threadMma1.partition_fragment_B(sVt);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;

    int n_block = n_block_max - 1;

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.barrier_Q.try_wait(0));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(0);
    }

    Tensor tSrS =
        partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    warp_scheduler_barrier_sync();
    gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
    warp_scheduler_barrier_arrive();
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read_k);
    ++smem_pipe_read_k;

    int mask_start_idx;
    int mask_row_id;
    int col_base;

    if constexpr (NeedMask) {
      const int lane_id = thread_idx % 32;
      mask_start_idx = mask[0] / kBlockN - 1;

      mask_row_id = thread_idx / 32 * 16 + lane_id / 4;

      col_base = thread_idx % 4 * 2;

      app_mask(tSrS, mask, mask_row_id, col_base + n_block * kBlockN);
    } else {
      auto col_limit_causal = [&](int row, int n_block) {
        return row + 1 + seq_len_k - n_block * kBlockN - seq_len_q +
               m_block * kBlockM;
      };
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
      Tensor tScS = threadMma0.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        if (int(get<1>(tScS(i))) >=
            std::min(seq_len_k - n_block * kBlockN,
                     col_limit_causal(int(get<0>(tScS(i))), n_block))) {
          tSrS(i) = -INFINITY;
        }
      }
    }

    softmax.template online_softmax</*Is_first=*/true>(
        tSrS, mainloop_params.softmax_scale_log2);

    Tensor tOrP = make_tensor(
        convert_type<Element>(tSrS).data(),
        convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
    Tensor scores_scale = make_fragment_like(softmax.row_max);
    clear(scores_scale);

#pragma unroll 2
    for (; n_block > 0; --n_block) {
      Tensor tSrS =
          partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      consumer_wait(pipeline_k, smem_pipe_read_k);
      warp_scheduler_barrier_sync();

      if constexpr (NeedMask) {
        if (n_block >= mask_start_idx) {
          app_mask(tSrS, mask, mask_row_id, col_base + n_block * kBlockN);
        }
      }

      gemm</*zero_init=*/true, /*wg_wait=*/-1>(
          tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      softmax.rescale_o(tOrO, scores_scale);
      consumer_wait(pipeline_v, smem_pipe_read_v);
      gemm</*zero_init=*/false, /*wg_wait=*/-1>(
          tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read_k);  // release K
      cute::copy(softmax.template max</*Is_first=*/false>(
                     tSrS, mainloop_params.softmax_scale_log2),
                 scores_scale);
      softmax.template online_softmax</*Is_first=*/false>(
          tSrS, mainloop_params.softmax_scale_log2);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v);  // release V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
      cute::copy(
          make_tensor(convert_type<Element>(tSrS).data(),
                      convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(
                          tSrS.layout())),
          tOrP);
    }

    softmax.rescale_o(tOrO, scores_scale);
    consumer_wait(pipeline_v, smem_pipe_read_v);

    gemm</*zero_init=*/false, /*wg_wait=*/-1>(
        tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
    cute::copy(softmax.finalize(mainloop_params.softmax_scale_log2),
               scores_scale);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);
    ++smem_pipe_read_v;

    softmax.rescale_o(tOrO, scores_scale);
  }

  template <int NumMmaThreads,
            typename SharedStorage,
            typename FrgTensorO,
            typename TiledMma,
            typename T>
  CUTLASS_DEVICE void store(Params const& mainloop_params,
                            FrgTensorO const& tOrO,
                            SharedStorage& shared_storage,
                            TiledMma tiled_mma,
                            int thread_idx,
                            const int o_head_stride,
                            const int real_seq,
                            T* out_ptr) {
    Tensor sO =
        make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOrO_out = convert_type<output_type>(tOrO);
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    cutlass::arch::NamedBarrier::sync(
        NumMmaThreads, static_cast<int>(AttnNamedBarriers::ValueEmpty) /*id*/);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible
                                               // to TMA
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    Tensor gO = make_tensor(make_gmem_ptr(out_ptr),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(o_head_stride, _1{}));

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});

    Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

    if (real_seq >= kBlockM) {
      copy<true>(gmem_tiled_copy_O, tOsO, tOgO, tOcO);
    } else {
      copy<false>(gmem_tiled_copy_O, tOsO, tOgO, tOcO, real_seq);
    }
  }
};
