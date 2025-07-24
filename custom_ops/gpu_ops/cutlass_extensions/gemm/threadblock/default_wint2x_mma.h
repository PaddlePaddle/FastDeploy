/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cutlass_extensions/arch/mma.h"
#include "cutlass_extensions/gemm/threadblock/default_dq_mma.h"
#include "cutlass_extensions/gemm/threadblock/default_mma_core.h"
#include "cutlass_extensions/gemm/threadblock/wint2x_mma_multistage.h"
#include "cutlass_extensions/gemm/threadblock/wint2x_params_accessor.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadblockShape, typename ElementT, int GroupSize>
struct DefaultQuantParamsIterators {
private:
    static constexpr int kAlignment = 128 / sizeof_bits<ElementT>::value;
    static_assert((ThreadblockShape::kN % kAlignment) == 0, "");

    static constexpr int kRows =
        (GroupSize == -1) ? 1 : (ThreadblockShape::kK + GroupSize - 1) / GroupSize;
    static constexpr int kColumns = ThreadblockShape::kN;

    using IteratorThreadMap = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<kColumns, kRows>,
        kColumns / kAlignment, kAlignment>;

public:
    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
        MatrixShape<kRows, kColumns>, ElementT, layout::RowMajor, 0,
        IteratorThreadMap, kAlignment>;
    using SmemIterator = Iterator;
};

template <typename ThreadblockShape, int GroupSize>
struct DefaultQuantParamsIterators<ThreadblockShape, uint4b_t, GroupSize> {
private:
    static constexpr int kAlignment = 32 / sizeof_bits<uint4b_t>::value;
    static_assert((ThreadblockShape::kN % kAlignment) == 0, "");

    static constexpr int kRows =
        (GroupSize == -1) ? 1 : (ThreadblockShape::kK + 2 * GroupSize - 1) / (2 * GroupSize);
    static constexpr int kColumns =
        (GroupSize == -1) ? ThreadblockShape::kN : ThreadblockShape::kN * 2;

    using IteratorThreadMap = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<kColumns, kRows>,
        kColumns / kAlignment, kAlignment>;

public:
    using AccessType = cutlass::Array<uint4b_t, kAlignment>;
    using Iterator = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        MatrixShape<kRows, kColumns>, uint4b_t, layout::RowMajor,
        0, IteratorThreadMap, AccessType>;

    using SmemIterator = Iterator;
};

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone>
struct DefaultWint2xMma;

////////////////////////////////////////////////////////////////////////////////

template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    /// Operator performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultWint2xMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator,
    layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    kStages, Operator, SharedMemoryClear>
{
public:
    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
        "Element A must be fp16 or bf16");

    static_assert(platform::is_same<ElementB, uint2b_t>::value,
        "Element B must be uint2b_t");

    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
        "Mma multistage must dequantize after ldsm");

    using ElementSuperScale = ElementA;
    using ElementLocalScale = uint4b_t;
    using ElementCodeScaleZp = float;

    static constexpr int kGroupSize = 64;

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        std::max(kStages, 3), Operator, false, CacheOpA, CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, ElementA, LayoutA, 1, ThreadMapA,
        AccessTypeA>;

private:
    static constexpr int kColumnsInterleaved = LayoutB::kColumnsInterleaved;
    static constexpr int kRowsPerTile = LayoutB::kRowsPerTile;
    static_assert(!(MmaCore::Shape::kN % kColumnsInterleaved), "ThreadblockShape must be disivle by kColumnsInterleaved");
    static_assert(kRowsPerTile == MmaCore::Shape::kK, "");

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using WarpArrangement = typename ThreadMapB::Detail::WarpThreadArrangement;
    static_assert(!(WarpArrangement::kStrided % kColumnsInterleaved), "");

    using IteratorShapeB = MatrixShape<
        MmaCore::Shape::kK * kColumnsInterleaved, MmaCore::Shape::kN / kColumnsInterleaved>;
    using InterleavedThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<IteratorShapeB::kRow, IteratorShapeB::kColumn>,
        ThreadMapB::kThreads,
        layout::PitchLinearShape<WarpArrangement::kContiguous * kColumnsInterleaved,
            WarpArrangement::kStrided / kColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // Define iterators over tiles from the B operand
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        IteratorShapeB, ElementB, layout::ColumnMajor, 0, InterleavedThreadMapB,
        AccessTypeB>;

private:
    // Define iterators over tiles from extra quant params for B operand
    using IteratorSuperScale = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementSuperScale, -1>::Iterator;
    using SmemIteratorSuperScale = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementSuperScale, -1>::SmemIterator;

    using IteratorLocalScale = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementLocalScale, kGroupSize>::Iterator;
    using SmemIteratorLocalScale = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementLocalScale, kGroupSize>::SmemIterator;

    using IteratorCodeScaleZp = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementCodeScaleZp, -1>::Iterator;
    using SmemIteratorCodeScaleZp = typename DefaultQuantParamsIterators<
        ThreadblockShape, ElementCodeScaleZp, -1>::Iterator;

public:
    using QuantParamsAccessor = Wint2ParamsAccessor<
        ElementA, ThreadblockShape, IteratorSuperScale, SmemIteratorSuperScale,
        IteratorLocalScale, SmemIteratorLocalScale,
        IteratorCodeScaleZp, SmemIteratorCodeScaleZp, kStages, kGroupSize>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::Wint2xMmaMultistage<
        typename MmaCore::Shape,
        IteratorA, typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA,
        IteratorB, typename MmaCore::SmemIteratorB, MmaCore::kCacheOpB,
        ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy,
        kStages, QuantParamsAccessor, SharedMemoryClear>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
