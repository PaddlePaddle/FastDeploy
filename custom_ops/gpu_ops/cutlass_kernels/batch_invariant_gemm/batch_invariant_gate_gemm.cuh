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

#include <chrono>
#include <cstdio>

// clang-format off
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_helper.h"
#include "helper.h"
#include "batch_invariant_gemm/gate_gemm_streamk_kernel.cuh"
// clang-format on

namespace fastdeploy {

// Cooperative kernel for batch-invariant Gate GEMM on SM90.
//
// Uses KernelTmaWarpSpecializedCooperative (1 Producer + 2 Consumer WGs
// = 384 threads). Higher throughput per CTA due to 2 consumer WGs,
// but requires M_tile >= 128.
//
// A = [M, K] row-major (activations)
// B = [N, K] column-major (pre-transposed gate weight)
// D = [M, N] row-major (output)
// Bias = optional [N] vector added to each row
template <typename ElementAB_,
          typename ElementD_,
          bool HasBias,
          typename TileShape,
          typename ClusterShape>
struct GateGemmSm90 {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc = float;
  static constexpr bool kHasBias = HasBias;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementC = std::conditional_t<HasBias, ElementD, void>;
  using LayoutC = LayoutD;
  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideC = StrideD;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      HasBias ? (128 / cutlass::sizeof_bits<ElementD>::value) : 4;

  // Cooperative schedule — requires M_tile >= 128
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using FusionOp =
      std::conditional_t<HasBias,
                         cutlass::epilogue::fusion::LinCombEltAct<
                             cutlass::epilogue::thread::Identity,
                             ElementD,
                             float,
                             ElementC,
                             float,
                             cutlass::FloatRoundStyle::round_to_nearest>,
                         cutlass::epilogue::fusion::LinearCombination<
                             ElementD,
                             float,
                             void,
                             float,
                             cutlass::FloatRoundStyle::round_to_nearest>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc,
          float,
          ElementC,
          LayoutC,
          AlignmentCD,
          ElementD,
          LayoutD,
          AlignmentCD,
          EpilogueSchedule,
          FusionOp>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutA,
          AlignmentAB,
          ElementAB,
          LayoutB,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              CEStorageSize)>,
          KernelSchedule>::CollectiveOp;

  // Standard CUTLASS GemmUniversal + StreamKScheduler
  using KernelType = enable_sm90_or_later<
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::StreamKScheduler>>;

  struct GemmKernel : public KernelType {};
};

// Non-Cooperative StreamK kernel for batch-invariant Gate GEMM on SM90.
//
// Uses KernelTmaWarpSpecialized (1 Producer + 1 Consumer WG = 256 threads)
// with persistent StreamK loop for SplitK + Deterministic reduction.
// Removes the M_tile >= 128 constraint, enabling TileShape<64, 32, 64>.
template <typename ElementAB_,
          typename ElementD_,
          bool HasBias,
          typename TileShape,
          typename ClusterShape>
struct GateGemmSm90StreamK {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc = float;
  static constexpr bool kHasBias = HasBias;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementC = std::conditional_t<HasBias, ElementD, void>;
  using LayoutC = LayoutD;
  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideC = StrideD;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      HasBias ? (128 / cutlass::sizeof_bits<ElementD>::value) : 4;

  // Non-Cooperative schedule — no M_tile >= 128 constraint
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

  using FusionOp =
      std::conditional_t<HasBias,
                         cutlass::epilogue::fusion::LinCombEltAct<
                             cutlass::epilogue::thread::Identity,
                             ElementD,
                             float,
                             ElementC,
                             float,
                             cutlass::FloatRoundStyle::round_to_nearest>,
                         cutlass::epilogue::fusion::LinearCombination<
                             ElementD,
                             float,
                             void,
                             float,
                             cutlass::FloatRoundStyle::round_to_nearest>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc,
          float,
          ElementC,
          LayoutC,
          AlignmentCD,
          ElementD,
          LayoutD,
          AlignmentCD,
          EpilogueSchedule,
          FusionOp>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutA,
          AlignmentAB,
          ElementAB,
          LayoutB,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              CEStorageSize)>,
          KernelSchedule>::CollectiveOp;

  // Custom kernel: Non-Cooperative structure + StreamK persistent loop
  using KernelType =
      GemmWarpSpecializedStreamK<cute::Shape<int, int, int, int>,
                                 CollectiveMainloop,
                                 CollectiveEpilogue,
                                 cutlass::gemm::StreamKScheduler>;

  struct GemmKernel : public KernelType {};
};

// Host timing helper — enabled by CUTLASS_HOST_TIMING=1
inline bool host_timing_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("CUTLASS_HOST_TIMING");
    return env && std::string(env) == "1";
  }();
  return enabled;
}

// Launch cache switch — enabled by CUTLASS_LAUNCH_CACHE=1.
// Caches sm_count, splits, workspace, and skips repeated can_implement checks.
// Saves ~1.5-2μs per call. Incompatible with benchmark sweeps that change
// env vars at runtime (CUTLASS_GATE_GEMM_OPT / CUTLASS_GATE_GEMM_SPLITS).
inline bool launch_cache_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("CUTLASS_LAUNCH_CACHE");
    return env && std::string(env) == "1";
  }();
  return enabled;
}

// Per-template-instantiation launch cache (like cuBLAS handle).
// One instance per Gemm type, lives for the process lifetime.
template <typename GemmOp>
struct LaunchCache {
  GemmOp gemm_op;
  void *workspace_ptr = nullptr;
  size_t workspace_capacity = 0;
  int sm_count = 0;
  int splits = -1;  // cached for (N, K) — M-independent
  int cached_N = 0;
  int cached_K = 0;
  bool verified = false;  // can_implement passed

  static LaunchCache &instance() {
    static LaunchCache cache;
    return cache;
  }

  // Grow-only workspace: only re-allocate when needed size exceeds capacity.
  void ensure_workspace(size_t needed, paddle::Tensor const &ref_tensor) {
    if (needed <= workspace_capacity) return;
    if (workspace_ptr) cudaFree(workspace_ptr);
    cudaMalloc(&workspace_ptr, needed);
    workspace_capacity = needed;
  }

  int get_sm_count() {
    if (sm_count == 0) {
      sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    }
    return sm_count;
  }
};

// Build CUTLASS args (shared by both cached and uncached paths).
template <typename Gemm>
typename Gemm::GemmKernel::Arguments build_gate_gemm_args(
    paddle::Tensor &out,
    paddle::Tensor const &a,
    paddle::Tensor const &b,
    void const *bias_ptr,
    int M,
    int N,
    int K,
    int sm_count) {
  using namespace cute;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  auto a_ptr = static_cast<ElementAB const *>(a.data());
  auto b_ptr = static_cast<ElementAB const *>(b.data());
  auto d_ptr = static_cast<ElementD *>(const_cast<void *>(out.data()));

  typename GemmKernel::MainloopArguments mainloop_args{
      const_cast<ElementAB *>(a_ptr),
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1)),
      const_cast<ElementAB *>(b_ptr),
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1))};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {1.0f},
      d_ptr,
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1)),
      d_ptr,
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1))};

  if constexpr (Gemm::kHasBias) {
    using ElementC = typename Gemm::ElementC;
    epilogue_args.thread.beta = 1.0f;
    epilogue_args.ptr_C = static_cast<ElementC const *>(bias_ptr);
    epilogue_args.dC = StrideC{0, cute::Int<1>{}, 0};
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;

  return {cutlass::gemm::GemmUniversalMode::kGemm,
          {M, N, K, 1},
          mainloop_args,
          epilogue_args,
          hw_info};
}

// Cached env var for CUTLASS_GATE_GEMM_SPLITS — read once per process.
// Returns the env value or -1 if not set.
inline int cached_gate_gemm_splits_env() {
  static const int val = [] {
    const char *env = std::getenv("CUTLASS_GATE_GEMM_SPLITS");
    return env ? std::atoi(env) : -1;
  }();
  return val;
}

// Compute SplitK splits (M-independent for batch invariance).
// Heuristic: only add SplitK when n_tiles alone cannot fill SMs.
// When n_tiles >= 80% of sm_count, SplitK reduce overhead > utilization gain.
template <typename GemmKernel>
int compute_splits(int N, int K, int sm_count) {
  using TileShapeType = typename GemmKernel::TileShape;
  constexpr int N_TILE = cute::size<1>(TileShapeType{});
  constexpr int K_TILE = cute::size<2>(TileShapeType{});
  int k_iters = (K + K_TILE - 1) / K_TILE;

  // Use cached env read when launch cache is enabled; otherwise read every
  // call to support runtime switching during benchmark sweeps.
  int env_val = launch_cache_enabled() ? cached_gate_gemm_splits_env() : [&] {
    const char *e = std::getenv("CUTLASS_GATE_GEMM_SPLITS");
    return e ? std::atoi(e) : -1;
  }();
  if (env_val > 0) {
    return std::max(1, std::min(env_val, k_iters));
  }
  int n_tiles = (N + N_TILE - 1) / N_TILE;
  // When n_tiles already saturate SMs (>= 80%), splits=1 avoids reduce cost.
  if (n_tiles * 5 >= sm_count * 4) {
    return 1;
  }
  constexpr int MAX_AUTO_SPLITS = 8;
  int auto_splits = std::max(1, (sm_count + n_tiles - 1) / n_tiles);
  return std::min({auto_splits, k_iters, MAX_AUTO_SPLITS});
}

// Set SplitK + Deterministic scheduler on args.
template <typename GemmKernel>
void set_splitk_scheduler(typename GemmKernel::Arguments &args, int splits) {
  using StreamKParams =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
  args.scheduler.decomposition_mode = StreamKParams::DecompositionMode::SplitK;
  args.scheduler.reduction_mode = StreamKParams::ReductionMode::Deterministic;
  args.scheduler.splits = splits;
}

// Launch helper: allocates workspace and runs the CUTLASS kernel.
// When CUTLASS_LAUNCH_CACHE=1, caches sm_count/splits/workspace/can_implement
// across calls, saving ~1.5-2μs. Safe for production (single-thread per GPU).
template <typename Gemm>
void launch_gate_gemm(
    paddle::Tensor &out,      // [M, N] pre-allocated
    paddle::Tensor const &a,  // [M, K] row-major
    paddle::Tensor const &b,  // [N, K] col-major (transposed weight)
    void const *bias_ptr,     // nullptr if no bias
    int M,
    int N,
    int K) {
  using GemmKernel = typename Gemm::GemmKernel;
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  const bool timing = host_timing_enabled();
  const bool use_cache = launch_cache_enabled();
  auto t0 = std::chrono::high_resolution_clock::now();

  if (use_cache) {
    // === Cached path: reuse sm_count, splits, workspace, GemmOp ===
    auto &cache = LaunchCache<GemmOp>::instance();
    int sm_count = cache.get_sm_count();

    auto args =
        build_gate_gemm_args<Gemm>(out, a, b, bias_ptr, M, N, K, sm_count);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Splits: cache per (N, K) — M-independent for batch invariance
    if (cache.cached_N != N || cache.cached_K != K) {
      cache.splits = compute_splits<GemmKernel>(N, K, sm_count);
      cache.cached_N = N;
      cache.cached_K = K;
    }
    set_splitk_scheduler<GemmKernel>(args, cache.splits);

    auto t2 = std::chrono::high_resolution_clock::now();

    // can_implement: check once per template type
    if (!cache.verified) {
      CUTLASS_CHECK(cache.gemm_op.can_implement(args));
      cache.verified = true;
    }

    // Workspace: grow-only, bypass Paddle allocator
    size_t ws_needed = cache.gemm_op.get_workspace_size(args);
    cache.ensure_workspace(ws_needed, out);

    auto t5 = std::chrono::high_resolution_clock::now();

    auto stream = paddle::GetCurrentCUDAStream(out.place())->raw_stream();
    CUTLASS_CHECK(cache.gemm_op.initialize(args, cache.workspace_ptr, stream));

    auto t6 = std::chrono::high_resolution_clock::now();

    CUTLASS_CHECK(cache.gemm_op.run(stream));

    auto t7 = std::chrono::high_resolution_clock::now();

    if (timing) {
      auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
      };
      printf(
          "[CUTLASS cached] M=%d N=%d K=%d splits=%d\n", M, N, K, cache.splits);
      printf("  args build     : %7.1f us\n", us(t0, t1));
      printf("  splits+verify  : %7.1f us\n", us(t1, t2));
      printf("  workspace      : %7.1f us\n", us(t2, t5));
      printf("  initialize     : %7.1f us  (TMA descriptors)\n", us(t5, t6));
      printf("  run (launch)   : %7.1f us\n", us(t6, t7));
      printf("  TOTAL          : %7.1f us\n", us(t0, t7));
    }
    return;
  }

  // === Uncached path: original behavior, full timing breakdown ===
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  auto args = build_gate_gemm_args<Gemm>(
      out, a, b, bias_ptr, M, N, K, hw_info.sm_count);
  auto t1 = std::chrono::high_resolution_clock::now();

  int splits = compute_splits<GemmKernel>(N, K, hw_info.sm_count);
  set_splitk_scheduler<GemmKernel>(args, splits);

  auto t2 = std::chrono::high_resolution_clock::now();

  GemmOp gemm_op;

  auto t3 = std::chrono::high_resolution_clock::now();

  CUTLASS_CHECK(gemm_op.can_implement(args));

  auto t4 = std::chrono::high_resolution_clock::now();

  size_t workspace_size = gemm_op.get_workspace_size(args);
  phi::Allocator *allocator = paddle::GetAllocator(out.place());
  auto workspace = allocator->Allocate(workspace_size);

  auto t5 = std::chrono::high_resolution_clock::now();

  auto stream = paddle::GetCurrentCUDAStream(out.place())->raw_stream();
  cutlass::Status status = gemm_op.initialize(args, workspace->ptr(), stream);
  CUTLASS_CHECK(status);

  auto t6 = std::chrono::high_resolution_clock::now();

  status = gemm_op.run(stream);
  CUTLASS_CHECK(status);

  auto t7 = std::chrono::high_resolution_clock::now();

  if (timing) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    printf("[CUTLASS host timing] M=%d N=%d K=%d splits=%d\n", M, N, K, splits);
    printf("  args build     : %7.1f us  (stride+ptr+epilogue+hw_info)\n",
           us(t0, t1));
    printf("  splits calc    : %7.1f us  (env read + auto-splits)\n",
           us(t1, t2));
    printf("  GemmOp ctor    : %7.1f us\n", us(t2, t3));
    printf("  can_implement  : %7.1f us\n", us(t3, t4));
    printf("  workspace alloc: %7.1f us  (get_size + Allocate)\n", us(t4, t5));
    printf(
        "  initialize     : %7.1f us  (init_workspace + to_underlying_args + "
        "smem_attr)\n",
        us(t5, t6));
    printf("  run (launch)   : %7.1f us  (cuLaunchKernelEx)\n", us(t6, t7));
    printf("  TOTAL          : %7.1f us\n", us(t0, t7));
    printf("  ---\n");
    printf("  cacheable (ctor+can_impl+workspace): %7.1f us\n", us(t2, t5));
    printf("  TMA+smem (initialize):               %7.1f us\n", us(t5, t6));
  }
}

}  // namespace fastdeploy
