/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/arch/memory_copy_sm80.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_wint2x_dequantizer.h"
#include "cutlass_extensions/gemm/threadblock/wint2x_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Transform for input B applied in register after the LDS
    typename TransformBAfterLDS_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone>
class Wint2xMmaMultistage :
  public Wint2xMmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = Wint2xMmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;
  /// Transform for input B applied in register after the LDS
  using TransformBAfterLDS = TransformBAfterLDS_;

  static constexpr int kInterleave = IteratorB::Shape::kRow / Shape::kK;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  using LayoutScale = cutlass::layout::ColumnMajor;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
  using Dequantizer =
      warp::MmaTensorOpWin2xDequantizer<Operator,
                                        typename Base::WarpGemm,
                                        Operand::kB,
                                        typename WarpTransformedFragmentB::Element,
                                        cutlass::layout::ColumnMajor,
                                        32,
                                        WeightOnlyQuantOp::UNDEFINED>;
  static_assert(sizeof(Dequantizer) > 0, "Dequantizer template instantiation failed");

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:

  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };

  using ElementA = typename IteratorA::Element;
  using ElementB = typename IteratorB::Element;
  using LayoutDetailsForB = kernel::LayoutDetailsB<ElementA, ElementB, ArchTag>;

  static constexpr bool IsTileInterleaveLayout =
      layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;
  static_assert(!IsTileInterleaveLayout || (IsTileInterleaveLayout && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
      "Layout K must match threadblockK");

 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  // Wint2 unzip operator
  Dequantizer warp_dequantizer_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

  /// Transform for B in register
  TransformBAfterLDS transform_B_;

  uint8_t* smem_ptr_B_;
  uint8_t* ptr_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  Wint2xMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    CUTLASS_TRACE_DEVICE(" Shape: {%d, %d, %d}, IteratorB::Shape: {%d, %d}, kInterleave: %d",
        Shape::kM, Shape::kN, Shape::kK, IteratorB::Shape::kRow, IteratorB::Shape::kColumn, kInterleave);
    CUTLASS_TRACE_DEVICE(" kPartitionsK=%d, kWarpGemmIterations=%d, WarpCount={%d, %d}, warp_idx_m=%d, warp_idx_n=%d, warp_idx_k=%d",
        Policy::kPartitionsK, Base::kWarpGemmIterations,
        Base::WarpCount::kM, Base::WarpCount::kN, warp_idx_m, warp_idx_n, warp_idx_k);

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

    CUTLASS_TRACE_DEVICE(" Policy::SmemPaddingA: {%d, %d}; Policy::SmemPaddingB: {%d, %d}",
        Policy::SmemPaddingA::kRow, Policy::SmemPaddingA::kColumn, Policy::SmemPaddingB::kRow, Policy::SmemPaddingB::kColumn);
    CUTLASS_TRACE_DEVICE(" operand_A_ptr=%p, kRow=%d, kColumn=%d",
        shared_storage.operand_A.data(), static_cast<int>(Base::SharedStorage::ShapeA::kRow),
        static_cast<int>(Base::SharedStorage::ShapeA::kColumn));
    CUTLASS_TRACE_DEVICE(" operand_B_ptr=%p, kRow=%d, kColumn=%d, %d bytes; kElementsPerAccess=%d, sizeof(AccessType)=%d, AsyncCopyIterationsPerStageB=%d, kAccessesPerVector=%d",
        shared_storage.operand_B.data(),
        static_cast<int>(Base::SharedStorage::ShapeB::kRow), static_cast<int>(Base::SharedStorage::ShapeB::kColumn),
        static_cast<int>(sizeof(shared_storage.operand_B)),
        static_cast<int>(IteratorB::ThreadMap::kElementsPerAccess), static_cast<int>(sizeof(typename IteratorB::AccessType)),
        static_cast<int>(Detail::AsyncCopyIterationsPerStageB), static_cast<int>(IteratorB::kAccessesPerVector));

    smem_ptr_B_ = reinterpret_cast<uint8_t*>(shared_storage.operand_B.data());
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(IteratorA &iterator_A, IteratorB &iterator_B)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance_A(IteratorA &iterator_A, int group_start_A = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }
  }

  template <bool GlobalToSharedB>
  CUTLASS_DEVICE
  void copy_tiles_and_advance_B(IteratorB &iterator_B, int group_start_B = 0) {
    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          if (group_start_B == 0 && j == 0 && v == 0) {
            CUTLASS_TRACE_DEVICE(" dst_ptr=%p, iterator_B.get()=%p, kAccessesPerGroupB=%d, kAccessesPerVector=%d, sizeof(AccessType)=%d",
                reinterpret_cast<void*>(dst_ptr), reinterpret_cast<void*>(gmem_ptr),
                static_cast<int>(Detail::kAccessesPerGroupB), static_cast<int>(IteratorB::kAccessesPerVector),
                static_cast<int>(sizeof(typename IteratorB::Element)));
          }

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::copy_zfill<kSrcBytes, kCacheOpB, GlobalToSharedB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::copy<kSrcBytes, kCacheOpB, GlobalToSharedB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance_per_stage_A(IteratorA &iterator_A) {
    iterator_A.set_iteration_index(0);
    this->smem_iterator_A_.set_iteration_index(0);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      typename IteratorA::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorA::AccessType *>(
              this->smem_iterator_A_.get());

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        auto gmem_ptr = iterator_A.get();

        int const kSrcBytes =
            sizeof_bits<typename IteratorA::Element>::value *
            IteratorA::ThreadMap::kElementsPerAccess /
            IteratorA::kAccessesPerVector / 8;

        int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
            dst_ptr + v, iterator_A.get(), iterator_A.valid());

        ++iterator_A;
      }

      ++this->smem_iterator_A_;
    }
  }

  template <bool GlobalToSharedB, bool InitStage>
  CUTLASS_DEVICE
  void copy_tiles_and_advance_per_stage_B(IteratorB &iterator_B, int stage) {
    iterator_B.set_iteration_index(0);
    this->smem_iterator_B_.set_iteration_index(0);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      typename IteratorB::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorB::AccessType *>(
              this->smem_iterator_B_.get());

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        auto gmem_ptr = iterator_B.get();

        int const kSrcBytes =
            sizeof_bits<typename IteratorB::Element>::value *
            IteratorB::ThreadMap::kElementsPerAccess /
            IteratorB::kAccessesPerVector / 8;

        if (v == 0) {
          int gmem_offset = reinterpret_cast<int>(gmem_ptr) - reinterpret_cast<int>(ptr_B_);
          int gmem_k = 8192 * kInterleave / 4;
          int gmem_n = 1792 / kInterleave;
          int gmem_row = gmem_offset / gmem_k;
          int gmem_col = gmem_offset % gmem_k;

          int smem_offset = reinterpret_cast<int>(dst_ptr) - reinterpret_cast<int>(smem_ptr_B_);
          int smem_k = Shape::kK * kInterleave / 4;
          int smem_n = Shape::kN / kInterleave;
          int smem_row = smem_offset / smem_k;
          int smem_col = smem_offset % smem_k;

          uint8_t* gmem_uint8_ptr = reinterpret_cast<uint8_t*>(gmem_ptr);

          CUTLASS_TRACE_DEVICE(" [stage=%d] gmem_ptr=%p, smem_ptr=%p, bytes=%d; gmem: %dx%d, {%d, %d}, [%d, %d, %d, %d, %d, %d, %d, %d]; smem: {%d, %d};",
              stage, reinterpret_cast<void*>(gmem_ptr), reinterpret_cast<void*>(dst_ptr), kSrcBytes,
              gmem_n, gmem_k, gmem_row, gmem_col,
              static_cast<int>(gmem_uint8_ptr[0]), static_cast<int>(gmem_uint8_ptr[1]),
              static_cast<int>(gmem_uint8_ptr[2]), static_cast<int>(gmem_uint8_ptr[3]),
              static_cast<int>(gmem_uint8_ptr[4]), static_cast<int>(gmem_uint8_ptr[5]),
              static_cast<int>(gmem_uint8_ptr[6]), static_cast<int>(gmem_uint8_ptr[7]),
              smem_row, smem_col);
        }

        if (InitStage) {
          cutlass::arch::copy_zfill<kSrcBytes, kCacheOpB, GlobalToSharedB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());
        } else {
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::copy_zfill<kSrcBytes, kCacheOpB, GlobalToSharedB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::copy<kSrcBytes, kCacheOpB, GlobalToSharedB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }
        }

        ++iterator_B;
      }

      ++this->smem_iterator_B_;
    }
    __syncthreads();
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      // Async copy zipped B to shared memory.
      copy_tiles_and_advance_per_stage_A(iterator_A);

      // Async copy zipped B to shared memory.
      copy_tiles_and_advance_per_stage_B<true, true>(iterator_B, stage);

      // TODO: Async copy other quantized params to shared memory, local_scale, code_scale, code_zp, super_scale.

      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }

  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations, ///< [in|out] number of threadblock mainloop iterations remaining
    int stage)
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      // CUTLASS_TRACE_DEVICE(" [MMa] stage=%d, warp_mma_k=%d", stage, warp_mma_k);

      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragment from shared memory
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        int group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance_A(iterator_A, group_start_iteration_A);
        copy_tiles_and_advance_B<true>(iterator_B, group_start_iteration_B);
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance_A(iterator_A, group_start_iteration_A);
        copy_tiles_and_advance_B<true>(iterator_B, group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }
    }
  }

  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      IteratorB &iterator_B)
  {
#if 0
    int smem_k = Shape::kK * kInterleave / 4;
    int smem_n = Shape::kN / kInterleave;
    for (int i = 0; i < 3 * smem_n; ++i) {
      for (int j = 0; j < smem_k; ++j) {
        if (i % 3 == 0) {
          CUTLASS_TRACE_DEVICE(" [i=%d, j=%d, %dx%d] %d", i, j, smem_n, smem_k, static_cast<int>(smem_ptr_B_[i * smem_k + j]));
        }
      }
    }
#endif

    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == (-Base::kStages + 1));

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    if (PipeState::WarpLoadedFragmentA::kElements == 8) {
      ElementA* warp_frag_A_ptr = reinterpret_cast<ElementA*>(pipe_state.warp_loaded_frag_A_[0].data());
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_A_=[%f, %f, %f, %f, %f, %f, %f, %f], %d bytes",
          static_cast<float>(warp_frag_A_ptr[0]), static_cast<float>(warp_frag_A_ptr[1]),
          static_cast<float>(warp_frag_A_ptr[2]), static_cast<float>(warp_frag_A_ptr[3]),
          static_cast<float>(warp_frag_A_ptr[4]), static_cast<float>(warp_frag_A_ptr[5]),
          static_cast<float>(warp_frag_A_ptr[6]), static_cast<float>(warp_frag_A_ptr[7]),
          sizeof_bits<typename PipeState::WarpLoadedFragmentA>::value / 8);
    }
    if (PipeState::WarpLoadedFragmentB::kElements == 64) {
      uint8_t* reg_uint8_ptr = reinterpret_cast<uint8_t*>(pipe_state.warp_loaded_frag_B_[0].data());
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_B_=[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d], %d bytes",
          static_cast<int>(reg_uint8_ptr[0]), static_cast<int>(reg_uint8_ptr[1]),
          static_cast<int>(reg_uint8_ptr[2]), static_cast<int>(reg_uint8_ptr[3]),
          static_cast<int>(reg_uint8_ptr[4]), static_cast<int>(reg_uint8_ptr[5]),
          static_cast<int>(reg_uint8_ptr[6]), static_cast<int>(reg_uint8_ptr[7]),
          static_cast<int>(reg_uint8_ptr[8]), static_cast<int>(reg_uint8_ptr[9]),
          static_cast<int>(reg_uint8_ptr[10]), static_cast<int>(reg_uint8_ptr[11]),
          static_cast<int>(reg_uint8_ptr[12]), static_cast<int>(reg_uint8_ptr[13]),
          static_cast<int>(reg_uint8_ptr[14]), static_cast<int>(reg_uint8_ptr[15]),
          sizeof_bits<typename PipeState::WarpLoadedFragmentB>::value / 8);
    }

    typename TransformBAfterLDS::result_type unpacked_frag_B = transform_B_(pipe_state.warp_loaded_frag_B_[0]);
    if (TransformBAfterLDS::result_type::kElements == 64) {
      CUTLASS_TRACE_DEVICE(" TransformBAfterLDS::result_type::kElements: 64, %d bytes", sizeof_bits<typename TransformBAfterLDS::result_type>::value / 8);
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_B_[0:15]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
          static_cast<float>(unpacked_frag_B[0]), static_cast<float>(unpacked_frag_B[1]),
          static_cast<float>(unpacked_frag_B[2]), static_cast<float>(unpacked_frag_B[3]),
          static_cast<float>(unpacked_frag_B[4]), static_cast<float>(unpacked_frag_B[5]),
          static_cast<float>(unpacked_frag_B[6]), static_cast<float>(unpacked_frag_B[7]),
          static_cast<float>(unpacked_frag_B[8]), static_cast<float>(unpacked_frag_B[9]),
          static_cast<float>(unpacked_frag_B[10]), static_cast<float>(unpacked_frag_B[11]),
          static_cast<float>(unpacked_frag_B[12]), static_cast<float>(unpacked_frag_B[13]),
          static_cast<float>(unpacked_frag_B[14]), static_cast<float>(unpacked_frag_B[15]));
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_B_[16:31]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
          static_cast<float>(unpacked_frag_B[16]), static_cast<float>(unpacked_frag_B[17]),
          static_cast<float>(unpacked_frag_B[18]), static_cast<float>(unpacked_frag_B[19]),
          static_cast<float>(unpacked_frag_B[20]), static_cast<float>(unpacked_frag_B[21]),
          static_cast<float>(unpacked_frag_B[22]), static_cast<float>(unpacked_frag_B[23]),
          static_cast<float>(unpacked_frag_B[24]), static_cast<float>(unpacked_frag_B[25]),
          static_cast<float>(unpacked_frag_B[26]), static_cast<float>(unpacked_frag_B[27]),
          static_cast<float>(unpacked_frag_B[28]), static_cast<float>(unpacked_frag_B[29]),
          static_cast<float>(unpacked_frag_B[30]), static_cast<float>(unpacked_frag_B[31]));
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_B_[32:47]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
          static_cast<float>(unpacked_frag_B[32]), static_cast<float>(unpacked_frag_B[33]),
          static_cast<float>(unpacked_frag_B[34]), static_cast<float>(unpacked_frag_B[35]),
          static_cast<float>(unpacked_frag_B[36]), static_cast<float>(unpacked_frag_B[37]),
          static_cast<float>(unpacked_frag_B[38]), static_cast<float>(unpacked_frag_B[39]),
          static_cast<float>(unpacked_frag_B[40]), static_cast<float>(unpacked_frag_B[41]),
          static_cast<float>(unpacked_frag_B[42]), static_cast<float>(unpacked_frag_B[43]),
          static_cast<float>(unpacked_frag_B[44]), static_cast<float>(unpacked_frag_B[45]),
          static_cast<float>(unpacked_frag_B[46]), static_cast<float>(unpacked_frag_B[47]));
      CUTLASS_TRACE_DEVICE(" warp_loaded_frag_B_[48:63]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
          static_cast<float>(unpacked_frag_B[48]), static_cast<float>(unpacked_frag_B[49]),
          static_cast<float>(unpacked_frag_B[50]), static_cast<float>(unpacked_frag_B[51]),
          static_cast<float>(unpacked_frag_B[52]), static_cast<float>(unpacked_frag_B[53]),
          static_cast<float>(unpacked_frag_B[54]), static_cast<float>(unpacked_frag_B[55]),
          static_cast<float>(unpacked_frag_B[56]), static_cast<float>(unpacked_frag_B[57]),
          static_cast<float>(unpacked_frag_B[58]), static_cast<float>(unpacked_frag_B[59]),
          static_cast<float>(unpacked_frag_B[60]), static_cast<float>(unpacked_frag_B[61]),
          static_cast<float>(unpacked_frag_B[62]), static_cast<float>(unpacked_frag_B[63]));
    }

    typename Dequantizer::FragmentLocalScale warp_frag_local_scale;
    typename Dequantizer::FragmentCodeScale warp_frag_code_scale;
    typename Dequantizer::FragmentCodeZp warp_frag_code_zp;
    typename Dequantizer::FragmentSuperScale warp_frag_super_scale;
    typename Dequantizer::FragmentOutOperand warp_frag_out;

    CUTLASS_TRACE_DEVICE(" warp_dequantizer_ - start load");
    warp_dequantizer_.load(warp_frag_local_scale,
                           warp_frag_code_scale,
                           warp_frag_code_zp,
                           warp_frag_super_scale);

    CUTLASS_TRACE_DEVICE("warp_dequantizer_ - start dequant");
    warp_dequantizer_.dequantize(warp_frag_out,
                                 pipe_state.warp_loaded_frag_B_[0],
                                 warp_frag_local_scale,
                                 warp_frag_code_scale,
                                 warp_frag_code_zp,
                                 warp_frag_super_scale);

#if 0
    // Transform, if necessary, the first warp-tile's shared memory fragments
    warp_mma_.transform(
      pipe_state.warp_transformed_frag_A_[0],
      pipe_state.warp_transformed_frag_B_[0],
      pipe_state.warp_loaded_frag_A_[0],
      pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    int stage = Base::kStages - 1;

    // Mainloop
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A,
        iterator_B,
        gemm_k_iterations,
        stage);
      stage += 1;
      break;
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  #endif
  }

  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate, pre-load B to shared memory.
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    ptr_B_ = reinterpret_cast<uint8_t*>(iterator_B.get_origin_pointer());

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_B, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Initialize destination accumulators with source accumulators
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
