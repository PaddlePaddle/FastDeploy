/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/kernel/default_intA_nf4B_traits.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/threadblock/default_nf4_int8_interleaved_mma_multistage.h"

namespace cutlass {
namespace gemm {
namespace threadblock {
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
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false
    >
struct DefaulInt8Nf4InterleavedMma;

// int8 int8 int32 Gemm specialization. Author(zhengzekang)
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaulInt8Nf4InterleavedMma<int8_t,
                  LayoutA,
                  kAlignmentA,
                  int8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:

    using Mma = Int8Nf4InterleavedMma<int8_t,
                                   LayoutA,
                                   kAlignmentA,
                                   int8_t,
                                   LayoutB,
                                   kAlignmentB,
                                   ElementAccumulator,
                                   layout::RowMajor,
                                   arch::OpClassTensorOp,
                                   ArchTag,
                                   ThreadblockShape,
                                   WarpShape,
                                   InstructionShape,
                                   2,
                                   Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaulInt8Nf4InterleavedMma<int8_t,
                  LayoutA,
                  kAlignmentA,
                  int8_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:

    using Mma = Int8Nf4InterleavedMma<int8_t,
                                   LayoutA,
                                   kAlignmentA,
                                   int8_t,
                                   LayoutB,
                                   kAlignmentB,
                                   ElementAccumulator,
                                   layout::RowMajor,
                                   arch::OpClassTensorOp,
                                   ArchTag,
                                   ThreadblockShape,
                                   WarpShape,
                                   InstructionShape,
                                   kStages,
                                   Operator,
                                   SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};




// int8 int4 int32
template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator>
struct DefaulInt8Nf4InterleavedMma<int8_t,
                  LayoutA,
                  kAlignmentA,
                  cutlass::uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  2,
                  Operator> {

private:

    using Mma = Int8Nf4InterleavedMma<int8_t,
                                   LayoutA,
                                   kAlignmentA,
                                   cutlass::uint4b_t,
                                   LayoutB,
                                   kAlignmentB,
                                   ElementAccumulator,
                                   layout::RowMajor,
                                   arch::OpClassTensorOp,
                                   ArchTag,
                                   ThreadblockShape,
                                   WarpShape,
                                   InstructionShape,
                                   2,
                                   Operator>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

template<
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    ///
    int kStages,
    /// Shared memory clear option
    SharedMemoryClearOption SharedMemoryClear>
struct DefaulInt8Nf4InterleavedMma<int8_t,
                  LayoutA,
                  kAlignmentA,
                  cutlass::uint4b_t,
                  LayoutB,
                  kAlignmentB,
                  ElementAccumulator,
                  layout::RowMajor,
                  arch::OpClassTensorOp,
                  ArchTag,
                  ThreadblockShape,
                  WarpShape,
                  InstructionShape,
                  kStages,
                  Operator,
                  false,
                  SharedMemoryClear> {

private:

    using Mma = Int8Nf4InterleavedMma<int8_t,
                                   LayoutA,
                                   kAlignmentA,
                                   cutlass::uint4b_t,
                                   LayoutB,
                                   kAlignmentB,
                                   ElementAccumulator,
                                   layout::RowMajor,
                                   arch::OpClassTensorOp,
                                   ArchTag,
                                   ThreadblockShape,
                                   WarpShape,
                                   InstructionShape,
                                   kStages,
                                   Operator,
                                   SharedMemoryClear>;

public:
    // Define the MmaCore components
    using MmaCore = typename Mma::MmaCore;

    // Define iterators over tiles from the A operand
    using IteratorA = typename Mma::IteratorA;

    // Define iterators over tiles from the B operand
    using IteratorB = typename Mma::IteratorB;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = typename Mma::ThreadblockMma;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
