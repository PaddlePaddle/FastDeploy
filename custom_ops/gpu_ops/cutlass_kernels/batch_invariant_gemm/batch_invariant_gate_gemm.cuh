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

// Batch-invariant Gate GEMM kernel for MoE gate matmul on SM90.
//
// Uses CUTLASS 3.x PersistentScheduler to guarantee that output row i
// is bitwise identical regardless of batch size M.
//
// A = [M, K] row-major (activations)
// B = [N, K] column-major (pre-transposed gate weight)
// D = [M, N] row-major (output)
// Bias = optional [N] vector added to each row
template <typename ElementAB_,
          typename ElementD_,
          bool HasBias,
          typename TileShape,
          typename ClusterShape,
          typename KernelSchedule =
              cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          typename EpilogueSchedule =
              cutlass::epilogue::TmaWarpSpecializedCooperative>
struct GateGemmSm90 {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc = float;  // always fp32 accumulation for determinism
  static constexpr bool kHasBias = HasBias;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Bias element type matches output; void when no bias
  using ElementC = std::conditional_t<HasBias, ElementD, void>;
  using LayoutC = LayoutD;

  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideC = StrideD;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      HasBias ? (128 / cutlass::sizeof_bits<ElementD>::value) : 4;

  // Epilogue fusion: D = alpha * acc (+ bias if HasBias)
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

  // StreamKScheduler with SplitK decomposition + Deterministic reduce:
  // - SplitK: fixed K-splits per tile, K boundaries independent of M
  // - Deterministic: daisy-chain ordered accumulation
  // Together they guarantee batch invariance with much better SM utilization.
  using KernelType = enable_sm90_or_later<
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::StreamKScheduler>>;

  struct GemmKernel : public KernelType {};
};

// Variant using custom Non-Cooperative kernel with StreamK support.
// Removes the M_tile >= 128 constraint, enabling TileShape<64, 32, 64>.
// Uses KernelTmaWarpSpecialized (1 Producer + 1 Consumer WG = 256 threads)
// with persistent StreamK loop for SplitK + Deterministic reduction.
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

// Launch helper: allocates workspace and runs the CUTLASS kernel
template <typename Gemm>
void launch_gate_gemm(
    paddle::Tensor &out,      // [M, N] pre-allocated
    paddle::Tensor const &a,  // [M, K] row-major
    paddle::Tensor const &b,  // [N, K] col-major (transposed weight)
    void const *bias_ptr,     // nullptr if no bias
    int M,
    int N,
    int K) {
  using namespace cute;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
  StrideD d_stride =
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  auto a_ptr = static_cast<ElementAB const *>(a.data());
  auto b_ptr = static_cast<ElementAB const *>(b.data());
  auto d_ptr = static_cast<ElementD *>(const_cast<void *>(out.data()));

  typename GemmKernel::ProblemShape prob_shape{M, N, K, 1};

  typename GemmKernel::MainloopArguments mainloop_args{
      const_cast<ElementAB *>(a_ptr),
      a_stride,
      const_cast<ElementAB *>(b_ptr),
      b_stride};

  // Build epilogue arguments
  // For LinearCombination: alpha=1.0, beta=0 (no bias) or beta=1.0 (with bias)
  typename GemmKernel::EpilogueArguments epilogue_args{
      {1.0f},  // thread args (alpha)
      d_ptr,
      d_stride,
      d_ptr,
      d_stride};

  if constexpr (Gemm::kHasBias) {
    using ElementC = typename Gemm::ElementC;
    epilogue_args.thread.beta = 1.0f;
    epilogue_args.ptr_C = static_cast<ElementC const *>(bias_ptr);
    // Bias is [N] broadcast across M rows: M-stride=0, N-stride=1,
    // batch-stride=0
    epilogue_args.dC = StrideC{0, cute::Int<1>{}, 0};
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info};

  // SplitK + Deterministic reduce: K boundaries are fixed (M-independent),
  // daisy-chain accumulation ensures fixed FP order → batch invariance.
  //
  // Optimal splits from nsys benchmark (H800, K=7168, N=256):
  //   TileShape<128,32,64> (Cooperative):      splits=5 → 13.3μs
  //   TileShape<64,32,64>  (Non-Coop StreamK): splits=3 →  9.9μs
  // Override via CUTLASS_GATE_GEMM_SPLITS env var for tuning.
  using TileShapeType = typename GemmKernel::TileShape;
  constexpr int M_TILE = cute::size<0>(TileShapeType{});
  constexpr int DEFAULT_SPLITS = (M_TILE <= 64) ? 3 : 5;
  constexpr int K_TILE = cute::size<2>(TileShapeType{});
  int k_iters = (K + K_TILE - 1) / K_TILE;
  int splits;
  const char *env_splits = std::getenv("CUTLASS_GATE_GEMM_SPLITS");
  if (env_splits) {
    splits = std::max(1, std::min(std::atoi(env_splits), k_iters));
  } else {
    splits = std::min(DEFAULT_SPLITS, k_iters);
  }

  using StreamKParams =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
  args.scheduler.decomposition_mode = StreamKParams::DecompositionMode::SplitK;
  args.scheduler.reduction_mode = StreamKParams::ReductionMode::Deterministic;
  args.scheduler.splits = splits;

  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  phi::Allocator *allocator = paddle::GetAllocator(out.place());
  auto workspace = allocator->Allocate(workspace_size);

  auto stream = paddle::GetCurrentCUDAStream(out.place())->raw_stream();
  cutlass::Status status = gemm_op.run(args, workspace->ptr(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace fastdeploy
