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

#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/trace.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Original data type
    typename T,
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterators over super scales in global memory
    typename IteratorSuperScale_,
    /// Iterators over super scales in shared memory
    typename SmemIteratorSuperScale_,
    /// Iterators over local scales in global memory
    typename IteratorLocalScale_,
    /// Iterators over local scales in shared memory
    typename SmemIteratorLocalScale_,
    /// Iterators over code scales and zps in global memory
    typename IteratorCodeScaleZp_,
    /// Iterators over code scales and zps in shared memory
    typename SmemIteratorCodeScaleZp_,
    /// Number of stages,
    int Stages_,
    /// Group size for quantization
    int GroupSize_>
class Wint2ParamsAccessor {
public:
  static_assert(platform::is_same<T, half_t>::value || platform::is_same<T, bfloat16_t>::value,
        "T must be fp16 or bf16");

  using ElementType = T;
  using Shape = Shape_;

  using IteratorSuperScale = IteratorSuperScale_;
  using SmemIteratorSuperScale = SmemIteratorSuperScale_;

  using IteratorLocalScale = IteratorLocalScale_;
  using SmemIteratorLocalScale = SmemIteratorLocalScale_;

  using IteratorCodeScaleZp = IteratorCodeScaleZp_;
  using SmemIteratorCodeScaleZp = SmemIteratorCodeScaleZp_;

  constexpr static int kStages = Stages_;
  constexpr static int kGroupSize = GroupSize_;

  using ElementSuperScale = typename IteratorSuperScale::Element;
  using LayoutSuperScale = typename IteratorSuperScale::Layout;

  /// local_scale uint4 and group-wise
  using ElementLocalScale = typename IteratorLocalScale::Element;
  using LayoutLocalScale = typename IteratorLocalScale::Layout;
  static_assert(platform::is_same<ElementLocalScale, uint4b_t>::value,
        "local_scale's type must be uint4b_t.");

  using ElementCodeScaleZp = typename IteratorCodeScaleZp::Element;
  using LayoutCodeScaleZp = typename IteratorCodeScaleZp::Layout;

  /// 2 uint4b_t values are stored in a single uint8_t
  constexpr static int kStagesPerLocalScaleLoad = 2 * kGroupSize / Shape::kK;
  constexpr static int kLocalScaleRows =
      IteratorLocalScale::Shape::kRow * IteratorLocalScale::Shape::kColumn * sizeof_bits<ElementLocalScale>::value / 8 / Shape::kN;

  using SmemElement = uint8_t;
  constexpr static int kSmemRows =
      kLocalScaleRows * kStages + sizeof(ElementSuperScale) + sizeof(ElementCodeScaleZp) * 2;
  constexpr static int kSmemColumns = Shape::kN;

  using QuantParamsShape = MatrixShape<kSmemRows, kSmemColumns>;

  constexpr static int kSuperScaleSmemOffset = 0;
  constexpr static int kCodeScaleSmemOffset = kSmemColumns * sizeof(ElementSuperScale);
  constexpr static int kCodeZpSmemOffset = kCodeScaleSmemOffset + kSmemColumns * sizeof(ElementCodeScaleZp);
  constexpr static int kLocalScaleSmemOffset = kCodeZpSmemOffset + kSmemColumns * sizeof(ElementCodeScaleZp);

  /// TensorRef type for loading element from a tensor
  using SuperTensorRef = cutlass::TensorRef<ElementSuperScale, LayoutSuperScale>;
  using LocalTensorRef = cutlass::TensorRef<ElementLocalScale, LayoutLocalScale>;
  using CodeTensorRef = cutlass::TensorRef<ElementCodeScaleZp, LayoutCodeScaleZp>;

  struct Arguments {
    IteratorSuperScale iterator_super_scale;
    IteratorLocalScale iterator_local_scale;
    IteratorCodeScaleZp iterator_code_scale;
    IteratorCodeScaleZp iterator_code_zp;

    int local_scale_pointer_offset;

    CUTLASS_DEVICE
    Arguments(IteratorSuperScale iterator_super_scale,
              IteratorLocalScale iterator_local_scale,
              IteratorCodeScaleZp iterator_code_scale,
              IteratorCodeScaleZp iterator_code_zp,
              int local_scale_pointer_offset)
      : iterator_super_scale(iterator_super_scale),
        iterator_local_scale(iterator_local_scale),
        iterator_code_scale(iterator_code_scale),
        iterator_code_zp(iterator_code_zp),
        local_scale_pointer_offset(local_scale_pointer_offset) {}
  };

private:
  //
  // Data members
  //

  /// Begin address of shared memory
  uint8_t* smem_pointer_;

  /// Iterator to write threadblock-scoped tile of super scale operand to shared memory
  SmemIteratorSuperScale smem_iterator_super_scale_;
  /// Iterator to write threadblock-scoped tile of local scale operand to shared memory
  SmemIteratorLocalScale smem_iterator_local_scale_;
  /// Iterator to write threadblock-scoped tile of code scale operand to shared memory
  SmemIteratorCodeScaleZp smem_iterator_code_scale_;
  /// Iterator to write threadblock-scoped tile of code zp operand to shared memory
  SmemIteratorCodeScaleZp smem_iterator_code_zp_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

  CUTLASS_DEVICE
  ElementSuperScale* get_super_scale_smem_ptr() {
    return reinterpret_cast<ElementSuperScale*>(smem_pointer_ + kSuperScaleSmemOffset);
  }

  CUTLASS_DEVICE
  ElementLocalScale* get_local_scale_smem_ptr() {
    return reinterpret_cast<ElementLocalScale*>(smem_pointer_ + kLocalScaleSmemOffset);
  }

  CUTLASS_DEVICE
  ElementCodeScaleZp* get_code_scale_smem_ptr() {
    return reinterpret_cast<ElementCodeScaleZp*>(smem_pointer_ + kCodeScaleSmemOffset);
  }

  CUTLASS_DEVICE
  ElementCodeScaleZp* get_code_zp_smem_ptr() {
    return reinterpret_cast<ElementCodeScaleZp*>(smem_pointer_ + kCodeZpSmemOffset);
  }

public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  Wint2ParamsAccessor(
      ///< prointer of shared memory
      uint8_t* smem_pointer,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
    : smem_pointer_(smem_pointer),
      smem_iterator_super_scale_(LayoutSuperScale(IteratorSuperScale::Shape::kColumn),
          get_super_scale_smem_ptr(), {1, IteratorSuperScale::Shape::kColumn}, thread_idx),
      smem_iterator_local_scale_(LayoutLocalScale(IteratorLocalScale::Shape::kColumn),
          get_local_scale_smem_ptr(), {1, IteratorLocalScale::Shape::kColumn}, thread_idx),
      smem_iterator_code_scale_(LayoutCodeScaleZp(IteratorCodeScaleZp::Shape::kColumn),
          get_code_scale_smem_ptr(), {1, IteratorCodeScaleZp::Shape::kColumn}, thread_idx),
      smem_iterator_code_zp_(LayoutCodeScaleZp(IteratorCodeScaleZp::Shape::kColumn),
          get_code_zp_smem_ptr(), {1, IteratorCodeScaleZp::Shape::kColumn}, thread_idx),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0) {}

  CUTLASS_DEVICE
  SuperTensorRef super_scale_ref() {
    return {get_super_scale_smem_ptr(), LayoutSuperScale(IteratorSuperScale::Shape::kColumn)};
  }

  CUTLASS_DEVICE
  LocalTensorRef local_scale_ref() {
    return {get_local_scale_smem_ptr(), LayoutLocalScale(IteratorLocalScale::Shape::kColumn)};
  }

  CUTLASS_DEVICE
  CodeTensorRef code_scale_ref() {
    return {get_code_scale_smem_ptr(), LayoutCodeScaleZp(IteratorCodeScaleZp::Shape::kColumn)};
  }

  CUTLASS_DEVICE
  CodeTensorRef code_zp_ref() {
    return {get_code_zp_smem_ptr(), LayoutCodeScaleZp(IteratorCodeScaleZp::Shape::kColumn)};
  }

  template <bool IsFirstStage>
  CUTLASS_DEVICE
  void copy_tiles_and_advance_per_stage(Arguments &quant_args, int stage) {
    if constexpr (IsFirstStage) {
      // Load channel-wise super_scale to shared memory, which only needs to be done once.
      typename IteratorSuperScale::Fragment tb_frag_super_scale;
      tb_frag_super_scale.clear();
      quant_args.iterator_super_scale.load(tb_frag_super_scale);
      this->smem_iterator_super_scale_.store(tb_frag_super_scale);

      // Load channel-wise code_scale to shared memory, which only needs to be done once.
      typename IteratorCodeScaleZp::Fragment tb_frag_code_scale;
      tb_frag_code_scale.clear();
      quant_args.iterator_code_scale.load(tb_frag_code_scale);
      this->smem_iterator_code_scale_.store(tb_frag_code_scale);

      // Load channel-wise code_zp to shared memory, which only needs to be done once.
      typename IteratorCodeScaleZp::Fragment tb_frag_code_zp;
      tb_frag_code_zp.clear();
      quant_args.iterator_code_zp.load(tb_frag_code_zp);
      this->smem_iterator_code_zp_.store(tb_frag_code_zp);
    }

    if ((stage % kStagesPerLocalScaleLoad) == 0) {
      // Load group-wise local_scale to shared memory, which only needs to be done at each stage.
      // Since 2 uint4b_t values of local_scale are saved in a single uint8_t, local_scale needs to be loaded once every two stages.
      using AccessType = typename IteratorLocalScale::AccessType;
      cutlass::arch::CacheOperation::Kind const kCacheOp = (sizeof_bits<AccessType>::value == 128)
          ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

      quant_args.iterator_local_scale.set_iteration_index(0);
      this->smem_iterator_local_scale_.set_iteration_index(0);

      // Async Copy for local_scale
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IteratorLocalScale::ThreadMap::Iterations::kCount; ++j) {
        AccessType *dst_ptr =
            reinterpret_cast<AccessType *>(this->smem_iterator_local_scale_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorLocalScale::kAccessesPerVector; ++v) {
          auto gmem_ptr = quant_args.iterator_local_scale.get();

          int const kSrcBytes =
              sizeof_bits<typename IteratorLocalScale::Element>::value *
              IteratorLocalScale::ThreadMap::kElementsPerAccess /
              IteratorLocalScale::kAccessesPerVector / 8;

              cutlass::arch::cp_async<kSrcBytes, kCacheOp>(
                  dst_ptr + v, gmem_ptr, quant_args.iterator_local_scale.valid());
        }
        ++quant_args.iterator_local_scale;
      }
      ++this->smem_iterator_local_scale_;
    }
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage(Arguments &quant_args) {
    if (smem_write_stage_idx_ % kStagesPerLocalScaleLoad == 0) {
      // Advance global iterators
      quant_args.iterator_local_scale.add_pointer_offset(quant_args.local_scale_pointer_offset);

      // Advance shared iterators
      int smem_pointer_offset = IteratorLocalScale::Shape::kRow * IteratorLocalScale::Shape::kColumn;
      smem_iterator_local_scale_.add_pointer_offset(smem_pointer_offset);
    }

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == kStagesPerLocalScaleLoad * kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      int pointer_offset = - kStages * IteratorLocalScale::Shape::kRow * IteratorLocalScale::Shape::kColumn;
      smem_iterator_local_scale_.add_pointer_offset(pointer_offset);
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  int advance_smem_read_stage() {
    int byte_offset = 0;

    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ % kStagesPerLocalScaleLoad == 0) {
      byte_offset = kLocalScaleRows * kSmemColumns;
    }

    if (smem_read_stage_idx_ == kStagesPerLocalScaleLoad * kStages) {
      smem_read_stage_idx_ = 0;
      byte_offset = - (kStages - 1) * kLocalScaleRows * kSmemColumns;
    }

    return byte_offset;
  }

  CUTLASS_DEVICE
  int clear_mask(Arguments &quant_args, bool cond) {
    quant_args.iterator_local_scale.clear_mask(cond);
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
