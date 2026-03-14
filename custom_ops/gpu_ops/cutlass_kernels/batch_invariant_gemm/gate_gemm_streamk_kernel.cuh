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

// Custom CUTLASS kernel: Non-Cooperative WarpSpecialized structure
// (1 Producer + 1 Consumer WG = 256 threads) with StreamKScheduler
// for SplitK + Deterministic reduction.
//
// This bypasses the Cooperative kernel's M_tile >= 128 constraint,
// enabling smaller TileShapes like <64, 32, 64>.
//
// See docs/cutlass_non_cooperative_streamk_analysis.md for design details.

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/grid_dependency_control.h"
// clang-format on

namespace fastdeploy {

template <class ProblemShape_,
          class CollectiveMainloop_,
          class CollectiveEpilogue_,
          class TileScheduler_ = cutlass::gemm::StreamKScheduler>
class GemmWarpSpecializedStreamK {
 public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 ||
                    cute::rank(ProblemShape{}) == 4,
                "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  // Tile scheduler — StreamKScheduler for SplitK + Deterministic reduce
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
          TileSchedulerTag,
          ArchTag,
          TileShape,
          ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // 1 Producer WG + 1 Consumer WG (no M_tile >= 128 constraint)
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(cute::size(TiledMma{})) +
      (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumProducerThreads =
      CollectiveMainloop::NumProducerThreadEvents;

  // Persistent mode: struct (not union) — mainloop/epilogue need separate smem
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, cute::_1> {
      using MainloopPipelineStorage =
          typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage =
          typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;

    struct TensorStorage : cute::aligned_struct<128, cute::_1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  static Params to_underlying_arguments(Arguments const& args,
                                        void* workspace) {
    auto problem_shape = args.problem_shape;
    if constexpr (cutlass::gemm::kernel::detail::Has_SwapAB_v<
                      CollectiveMainloop>) {
      cute::get<0>(problem_shape) = cute::get<1>(args.problem_shape);
      cute::get<1>(problem_shape) = cute::get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          args.hw_info.device_id);
    }
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
    }
    cutlass::KernelHardwareInfo hw_info{
        args.hw_info.device_id, sm_count, max_active_clusters};

    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(
        args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,
                                              cutlass::MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset +=
        TileScheduler::template get_workspace_size<ProblemShape,
                                                   ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = cutlass::round_nearest(workspace_offset,
                                              cutlass::MinWorkspaceAlignment);

    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler =
        TileScheduler::to_underlying_arguments(problem_shape_MNKL,
                                               TileShape{},
                                               ClusterShape{},
                                               hw_info,
                                               args.scheduler,
                                               scheduler_workspace,
                                               NumEpilogueSubTiles);

    return {args.mode,
            problem_shape,
            CollectiveMainloop::to_underlying_arguments(
                args.problem_shape, args.mainloop, nullptr),
            CollectiveEpilogue::to_underlying_arguments(
                args.problem_shape, args.epilogue, epilogue_workspace),
            hw_info,
            scheduler,
            workspace};
  }

  static bool can_implement(Arguments const& args) {
    bool ok = (args.mode == cutlass::gemm::GemmUniversalMode::kGemm) ||
              (args.mode == cutlass::gemm::GemmUniversalMode::kBatched &&
               cute::rank(ProblemShape{}) == 4);
    ok &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    ok &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    ok &= TileScheduler::can_implement(args.scheduler);
    return ok;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t ws = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    ws += CollectiveEpilogue::get_workspace_size(args.problem_shape,
                                                 args.epilogue);
    ws = cutlass::round_nearest(ws, cutlass::MinWorkspaceAlignment);
    ws += TileScheduler::template get_workspace_size<ProblemShape,
                                                     ElementAccumulator>(
        args.scheduler,
        args.problem_shape,
        args.hw_info,
        NumMmaWarpGroups,
        NumEpilogueSubTiles);
    ws = cutlass::round_nearest(ws, cutlass::MinWorkspaceAlignment);
    return ws;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    cutlass::Status status = cutlass::Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    status = CollectiveEpilogue::initialize_workspace(
        args.problem_shape,
        args.epilogue,
        workspace_ptr + workspace_offset,
        stream,
        cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(
        args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,
                                              cutlass::MinWorkspaceAlignment);
    if (status != cutlass::Status::kSuccess) return status;

    status = TileScheduler::template initialize_workspace<ProblemShape,
                                                          ElementAccumulator>(
        args.scheduler,
        workspace_ptr + workspace_offset,
        stream,
        args.problem_shape,
        args.hw_info,
        NumMmaWarpGroups,
        NumEpilogueSubTiles,
        /*NumAccumulatorMtxs=*/1,
        cuda_adapter);
    return status;
  }

  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order =
        params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
            ? TileScheduler::RasterOrderOptions::AlongN
            : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler,
                                         params.problem_shape,
                                         TileShape{},
                                         ClusterShape{},
                                         params.hw_info,
                                         args);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define ENABLE_SM90_KERNEL_LEVEL 1
#endif
#if !defined(ENABLE_SM90_KERNEL_LEVEL)
    printf(
        "ERROR : Arch conditional MMA instruction used without targeting "
        "sm90a compute capability. Aborting.\n");
#else

    static_assert(cute::rank(StrideA{}) == 3,
                  "StrideA must be rank-3: [M, K, L].");
    static_assert(cute::rank(StrideB{}) == 3,
                  "StrideB must be rank-3: [N, K, L].");
    static_assert(cute::rank(StrideC{}) == 3,
                  "StrideC must be rank-3: [M, N, L].");
    static_assert(cute::rank(StrideD{}) == 3,
                  "StrideD must be rank-3: [M, N, L].");

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };
    enum class ProducerWarpRole {
      MainloopEpilogue = 0,
      Warp1 = 1,
      Warp2 = 2,
      Warp3 = 3
    };

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // TMA descriptor prefetch
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    CollectiveEpilogue collective_epilogue(params.epilogue,
                                           shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

    // Mainloop pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes =
        params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       ClusterShape{});

    // Epilogue load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count =
        cutlass::NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes =
          params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load,
                                      epi_load_pipeline_params);

    // Epilogue store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Pipeline states
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;
    typename MainloopPipeline::PipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();
    typename EpiLoadPipeline::PipelineState epi_load_pipe_producer_state =
        cutlass::make_producer_start_state<EpiLoadPipeline>();
    typename EpiStorePipeline::PipelineState epi_store_pipe_producer_state =
        cutlass::make_producer_start_state<EpiStorePipeline>();

    // Cluster sync
    auto cluster_wait_fn = []() {
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return []() { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return []() {};
      }
    }();

    auto problem_shape_MNKL =
        cute::append<4>(params.problem_shape, cute::Int<1>{});
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};

    TileScheduler scheduler{params.scheduler};
    typename TileScheduler::WorkTileInfo work_tile_info;

    CollectiveMainloop collective_mainloop;
    auto load_inputs =
        collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2);
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Wait for cluster
    cluster_wait_fn();

    // ===== Producer WG =====
    if (warp_group_role == WarpGroupRole::Producer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        cutlass::arch::wait_on_dependent_grids();

        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next, inc] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next;
            continue;
          }

          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
              work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start =
              TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(
              idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          collective_mainloop.load(params.mainloop,
                                   mainloop_pipeline,
                                   mainloop_pipe_producer_state,
                                   load_inputs,
                                   blk_coord,
                                   k_tile_iter,
                                   work_k_tile_count,
                                   lane_idx,
                                   block_rank_in_cluster,
                                   shared_storage.tensors.mainloop);
          mainloop_pipe_producer_state.advance(work_k_tile_count);

          // Epilogue load for final split only
          if (is_epi_load_needed && TileScheduler::compute_epilogue(
                                        work_tile_info, params.scheduler)) {
            __syncwarp();
            epi_load_pipe_producer_state = collective_epilogue.load(
                epi_load_pipeline,
                epi_load_pipe_producer_state,
                problem_shape_MNKL,
                blk_shape,
                blk_coord,
                tiled_mma,
                lane_idx,
                shared_storage.tensors.epilogue,
                work_tile_info.reduction_subtile_idx());
          }

          auto [next, inc] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next;
        }  // while

        collective_mainloop.load_tail(mainloop_pipeline,
                                      mainloop_pipe_producer_state);
        if (is_epi_load_needed) {
          collective_epilogue.load_tail(epi_load_pipeline,
                                        epi_load_pipe_producer_state);
        }
      }  // MainloopEpilogue warp
    }    // Producer WG

    // ===== Consumer WG =====
    else if (warp_group_role == WarpGroupRole::Consumer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

      bool do_store_tail = false;
      while (work_tile_info.is_valid()) {
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
            work_tile_info, problem_shape_MNKL, blk_shape);

        auto accumulators =
            partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));

        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          collective_mainloop.mma(mainloop_pipeline,
                                  mainloop_pipe_consumer_state,
                                  accumulators,
                                  work_k_tile_count,
                                  warp_group_thread_idx,
                                  shared_storage.tensors.mainloop,
                                  params.mainloop);

          collective_mainloop.mma_tail(mainloop_pipeline,
                                       mainloop_pipe_consumer_state,
                                       work_k_tile_count);
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // SplitK reduction across splits (no-op for non-split tiles)
        int consumer_warp_group_idx =
            cutlass::canonical_warp_group_idx() - NumLoadWarpGroups;
        TileScheduler::fixup(params.scheduler,
                             work_tile_info,
                             accumulators,
                             NumMmaWarpGroups,
                             consumer_warp_group_idx);

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          auto [epi_load_next, epi_store_next] =
              collective_epilogue.store(epi_load_pipeline,
                                        epi_load_pipe_consumer_state,
                                        epi_store_pipeline,
                                        epi_store_pipe_producer_state,
                                        problem_shape_MNKL,
                                        blk_shape,
                                        blk_coord,
                                        accumulators,
                                        tiled_mma,
                                        warp_group_thread_idx,
                                        shared_storage.tensors.epilogue,
                                        work_tile_info.reduction_subtile_idx());
          epi_load_pipe_consumer_state = epi_load_next;
          epi_store_pipe_producer_state = epi_store_next;
          do_store_tail = true;
        }

        auto [next, inc] = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = next;
      }  // while

      if (do_store_tail) {
        collective_epilogue.store_tail(epi_load_pipeline,
                                       epi_load_pipe_consumer_state,
                                       epi_store_pipeline,
                                       epi_store_pipe_producer_state);
      }
    }  // Consumer WG

#endif
  }
};

}  // namespace fastdeploy
