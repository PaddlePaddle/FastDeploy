/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/threadblock/nf4_int8_mma_base.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/threadblock/int8_mma_base.h"

#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/interleaved_numeric_conversion_nf4.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
[[gnu::warning("your type here")]]
bool print_type() { return false; }

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template<
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
    // global memory of iterator  NF4LookUpTable
    typename IteratorNF4LookUpTable_,
    // share memory iterator of NF4LookUpTable
    typename SmemIteratorNF4LookUpTable_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Converter for B matrix applited immediately after the LDS
    typename TransformBAfterLDS_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class Int8Nf4InterleavedMmaMultistage: public Int8Nf4InterleavedMmaBase<Shape_, Policy_, Stages> {
public:
    ///< Base class
    using Base = Int8Nf4InterleavedMmaBase<Shape_, Policy_, Stages>;
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape = Shape_;
    ///< Iterates over tiles of A operand in global memory
    using IteratorA = IteratorA_;
    ///< Iterates over tiles of B operand in global memory
    using IteratorB = IteratorB_;
    ///< Iterates over tiles of nf4 look up table in global memory
    using IteratorNF4LookUpTable = IteratorNF4LookUpTable_;
    using ElementNF4LookUpTable = typename IteratorNF4LookUpTable::Element;
    using LayoutNF4LookUpTable = typename IteratorNF4LookUpTable::Layout;


    ///< Data type of accumulator matrix
    using ElementC = ElementC_;
    ///< Layout of accumulator matrix
    using LayoutC = LayoutC_;
    ///< Policy describing tuning details
    using Policy = Policy_;


    using SmemIteratorA     = SmemIteratorA_;
    using SmemIteratorB     = SmemIteratorB_;
    using SmemIteratorNF4LookUpTable = SmemIteratorNF4LookUpTable_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    using TransformBAfterLDS = TransformBAfterLDS_;

    //
    // Dependent types
    //

    /// Fragment of accumulator tile
    using FragmentC = typename Policy::Operator::FragmentC;

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Minimum architecture is Sm80 to support cp.async
    using ArchTag = arch::Sm80;

    /// Complex transform on A operand
    static ComplexTransform const kTransformA = Operator::kTransformA;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB = Operator::kTransformB;

    /// Internal structure exposed for introspection.
    struct Detail {

        static_assert(Base::kWarpGemmIterations > 1,
                      "The pipelined structure requires at least two warp-level "
                      "GEMM operations.");
        // static_assert(Base::kWarpGemmIterations==4,"Base::kWarpGemmIterations!=4");
        /// Number of cp.async instructions to load one stage of operand A
        static int const AsyncCopyIterationsPerStageA = IteratorA::ThreadMap::Iterations::kCount;

        /// Number of cp.async instructions to load one stage of operand B
        static int const AsyncCopyIterationsPerStageB = IteratorB::ThreadMap::Iterations::kCount;

        /// Number of stages
        static int const kStages = Stages;

        /// Number of cp.async instructions to load on group of operand A
        static int const kAccessesPerGroupA =
            (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

        /// Number of cp.async instructions to load on group of operand B
        static int const kAccessesPerGroupB =
            (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
    };

private:
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;
    using ElementB          = typename IteratorB::Element;
    using LayoutDetailsForB = kernel::LayoutDetailsB<ElementB, ArchTag>;

    static constexpr bool RequiresTileInterleave =
        layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;
    static_assert(!RequiresTileInterleave || (RequiresTileInterleave && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
                  "Layout K must match threadblockK");

private:
    //
    // Data members
    //

    /// Iterator to write threadblock-scoped tile of A operand to shared memory
    SmemIteratorA smem_iterator_A_;

    /// Iterator to write threadblock-scoped tile of B operand to shared memory
    SmemIteratorB smem_iterator_B_;

    SmemIteratorNF4LookUpTable smem_iterator_nf4_look_up_table_;

public:
    /// Construct from tensor references
    CUTLASS_DEVICE
    Int8Nf4InterleavedMmaMultistage(
        ///< Shared storage needed for internal use by threadblock-scoped GEMM
        typename Base::SharedStorage& shared_storage,
        ///< ID within the threadblock
        int thread_idx,
        ///< ID of warp
        int warp_idx,
        ///< ID of each thread within a warp
        int lane_idx):
        Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_iterator_nf4_look_up_table_(LayoutNF4LookUpTable(16),
                                         shared_storage.operand_nf4_look_up_table.data(),
                                         {1, 16},
                                         thread_idx)
{
        // Compute warp location within threadblock tile by mapping the warp_id to
        // three coordinates:
        //   _m: the warp's position within the threadblock along the M dimension
        //   _n: the warp's position within the threadblock along the N dimension
        //   _k: the warp's position within the threadblock along the K dimension

        int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k  = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterationsForB * warp_idx_k, warp_idx_n});


        // if((threadIdx.x % 32) == 0){
        //     printf("#### %d-%d-%d-%d-%d-%d, gmem_ptr_nf4_look_up_table:%p, kSrcBytesNf4:%d \n",
        //             blockIdx.x, blockIdx.y, blockIdx.z,
        //             threadIdx.x, threadIdx.y, threadIdx.z,
        //             gmem_ptr_nf4_look_up_table,
        //             kSrcBytesNf4);
        // }
        // cutlass::arch::cp_async_zfill<kSrcBytesNf4, cutlass::arch::CacheOperation::Global>(
        //                 dst_ptr_nf4_look_up_table, gmem_ptr_nf4_look_up_table, iterator_nf4_look_up_table.valid());
    }

    CUTLASS_DEVICE
    void
    copy_tiles_and_advance(IteratorA& iterator_A, IteratorB& iterator_B, int group_start_A = 0, int group_start_B = 0)
    {
        iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
        this->smem_iterator_A_.set_iteration_index(group_start_A);

        // Async Copy for operand A
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
            if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
                typename IteratorA::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

                int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value
                                      * IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_A.get();

                    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
                        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                    }
                    else {
                        cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(dst_ptr + v, gmem_ptr, iterator_A.valid());
                    }

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }
        }

        iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
        this->smem_iterator_B_.set_iteration_index(group_start_B);

        // Async Copy for operand B
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
            if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
                typename IteratorB::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

                int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value
                                      * IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B.get();

                    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
                        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                    }
                    else {
                        cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(dst_ptr + v, gmem_ptr, iterator_B.valid());
                    }

                    // if(true && (threadIdx.x||threadIdx.y||threadIdx.z)==0){
                    //     int32_t* print_ptr = reinterpret_cast<int32_t*>(iterator_B.get());
                    //     int32_t* print_ptr_smem = reinterpret_cast<int32_t*>(dst_ptr+v);
                    //     if (iterator_B.valid())
                    //     {
                    //         printf("gmem_ptr cp source of thread %d-%d-%d;%d-%d-%d: %p:%x-%x-%x-%x=>%x,%x,%x,%x \n",
                    //             blockIdx.x,blockIdx.y,blockIdx.z,
                    //             threadIdx.x,threadIdx.y,threadIdx.z,
                    //             iterator_B.get(),
                    //             static_cast<uint32_t>(print_ptr[0]),
                    //             static_cast<uint32_t>(print_ptr[1]),
                    //             static_cast<uint32_t>(print_ptr[2]),
                    //             static_cast<uint32_t>(print_ptr[3]),
                    //             static_cast<uint32_t>(print_ptr_smem[0]),
                    //             static_cast<uint32_t>(print_ptr_smem[1]),
                    //             static_cast<uint32_t>(print_ptr_smem[2]),
                    //             static_cast<uint32_t>(print_ptr_smem[3]));
                    //     }
                    // }
                    ++iterator_B;
                }
                ++this->smem_iterator_B_;
            }
        }
    }

    /// Perform a threadblock-scoped matrix multiply-accumulate
    CUTLASS_DEVICE
    void operator()(
        ///< problem size of GEMM
        int gemm_k_iterations,
        ///< destination accumulator tile
        FragmentC& accum,
        ///< iterator over A operand in global memory
        IteratorA iterator_A,
        ///< iterator over B operand in global memory
        IteratorB iterator_B,
        IteratorNF4LookUpTable iterator_nf4_look_up_table,
        ///< initial value of accumulator
        FragmentC const& src_accum)
    {

        // printf("gemm_k_iterations:%d\n", gemm_k_iterations);

        //
        // Prologue
        //

        // use share memory to get look_up_table of nf4;

        // __shared__ uint32_t shared_look_up_table[16];

        // int lane_idx=threadIdx.x%32;
        // int warp_idx=threadIdx.x/32;
        // if(lane_idx<16){
        //     shared_look_up_table[lane_idx]=lane_idx;
        // }

        // __shared__ uint32_t shared_look_up_table[32][32];
        // if(warp_idx==0){
        //     CUTLASS_PRAGMA_UNROLL
        //     for(int ii=0;ii<16;++ii){
        //         shared_look_up_table[lane_idx][ii]=ii;
        //     }
        // }

        /// load look up table to smem here
        // __shared__ int32_t nf4_smem_look_up_table[16];

        // int32_t* gmem_ptr_nf4_look_up_table = reinterpret_cast<int32_t*>(iterator_nf4_look_up_table.get());
        // // smem look up table
        // int32_t* dst_ptr_nf4_look_up_table = reinterpret_cast<int32_t*>(nf4_smem_look_up_table);

        // if(lane_idx == 0){
        //     int4* dst_ptr_nf4_look_up_table_int4 = reinterpret_cast<int4*>(nf4_smem_look_up_table);
        //     dst_ptr_nf4_look_up_table_int4[lane_idx] = *(reinterpret_cast<int4*>(gmem_ptr_nf4_look_up_table) + lane_idx);
        // }
        // __syncthreads();
        // // reg look up table
        // cutlass::Array<uint32_t, 4, 1> reg_look_up_table;
        // CUTLASS_PRAGMA_UNROLL
        // for(int ii=0;ii<4;++ii){
        //     reg_look_up_table[ii]=*(reinterpret_cast<int32_t*>(dst_ptr_nf4_look_up_table) + ii);
        // }

        TransformBAfterLDS lds_converter;

        // NOTE - switch to ldg.sts
        // Issue this first, so cp.async.commit_group will commit this load as well.
        // Note: we do not commit here and this load will commit in the same group as
        //       the first load of A.

        // Issue several complete stages
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

            iterator_A.clear_mask(gemm_k_iterations == 0);
            iterator_B.clear_mask(gemm_k_iterations == 0);

            iterator_A.set_iteration_index(0);
            this->smem_iterator_A_.set_iteration_index(0);

            // Async Copy for operand A
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
                typename IteratorA::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorA::AccessType*>(this->smem_iterator_A_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value
                                          * IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector
                                          / 8;

                    int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                        dst_ptr + v, iterator_A.get(), iterator_A.valid());

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }

            iterator_B.set_iteration_index(0);
            // print_type<IteratorB>();
            this->smem_iterator_B_.set_iteration_index(0);

            // Async Copy for operand B
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
                typename IteratorB::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB::AccessType*>(this->smem_iterator_B_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
                    int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value
                                          * IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector
                                          / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                        dst_ptr + v, iterator_B.get(), iterator_B.valid());

                    // if(true && (threadIdx.x||threadIdx.y||threadIdx.z)==0){
                    //     int32_t* print_ptr = reinterpret_cast<int32_t*>(iterator_B.get());
                    //     int32_t* print_ptr_smem = reinterpret_cast<int32_t*>(dst_ptr+v);
                    //     if (iterator_B.valid())
                    //     {
                    //         printf("gmem_ptr cp source of thread %d-%d-%d;%d-%d-%d: %p:%x-%x-%x-%x=>%x,%x,%x,%x \n",
                    //             blockIdx.x,blockIdx.y,blockIdx.z,
                    //             threadIdx.x,threadIdx.y,threadIdx.z,
                    //             iterator_B.get(),
                    //             static_cast<uint32_t>(print_ptr[0]),
                    //             static_cast<uint32_t>(print_ptr[1]),
                    //             static_cast<uint32_t>(print_ptr[2]),
                    //             static_cast<uint32_t>(print_ptr[3]),
                    //             static_cast<uint32_t>(print_ptr_smem[0]),
                    //             static_cast<uint32_t>(print_ptr_smem[1]),
                    //             static_cast<uint32_t>(print_ptr_smem[2]),
                    //             static_cast<uint32_t>(print_ptr_smem[3]));
                    //     }
                    // }
                    ++iterator_B;
                }

                ++this->smem_iterator_B_;
            }

            // Move to the next stage
            iterator_A.add_tile_offset({0, 1});
            iterator_B.add_tile_offset({1, 0});

            this->smem_iterator_A_.add_tile_offset({0, 1});
            this->smem_iterator_B_.add_tile_offset({1, 0});

            // Defines the boundary of a stage of cp.async.
            cutlass::arch::cp_async_fence();
        }

        // Perform accumulation in the 'd' output operand
        accum = src_accum;

        //
        // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
        // so that all accumulator elements outside the GEMM footprint are zero.
        //

        if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

            /// Iterator to write threadblock-scoped tile of A operand to shared memory
            SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

            typename IteratorA::AccessType zero_A;
            zero_A.clear();

            last_smem_iterator_A.set_iteration_index(0);

            // Async Copy for operand A
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

                typename IteratorA::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorA::AccessType*>(last_smem_iterator_A.get());

                *dst_ptr = zero_A;

                ++last_smem_iterator_A;
            }

            /// Iterator to write threadblock-scoped tile of B operand to shared memory
            SmemIteratorB                  last_smem_iterator_B(this->smem_iterator_B_);
            typename IteratorB::AccessType zero_B;

            zero_B.clear();
            last_smem_iterator_B.set_iteration_index(0);

            // Async Copy for operand B
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

                typename IteratorB::AccessType* dst_ptr =
                    reinterpret_cast<typename IteratorB::AccessType*>(last_smem_iterator_B.get());

                *dst_ptr = zero_B;

                ++last_smem_iterator_B;
            }
        }

        // Waits until kStages-2 stages have committed.
        cutlass::arch::cp_async_wait<Base::kStages - 2>();
        __syncthreads();

        // Pair of fragments used to overlap shared memory loads and math
        // instructions
        WarpFragmentA                       warp_frag_A[2];
        WarpFragmentB                       warp_frag_B[2];
        typename TransformBAfterLDS::result_type converted_frag_B_buffer[2];
        Operator warp_mma;

        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.set_kgroup_index(0);

        this->warp_tile_iterator_B_.load(warp_frag_B[0]);
        converted_frag_B_buffer[0] =
            lds_converter(warp_frag_B[0]);
        this->warp_tile_iterator_A_.load(warp_frag_A[0]);

        // if((threadIdx.x||threadIdx.y||threadIdx.z)==0){
        //     uint32_t* frag_b_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_B[0]);
        //     printf("#### warp_frag_b_load [0] bid:%d-%d-%d,"
        //                 " frag_b_reg_ptr:%x-%x-%x-%x-%x-%x-%x-%x  \n",
        //             blockIdx.x,blockIdx.y,blockIdx.z,
        //             frag_b_reg_ptr[0],
        //             frag_b_reg_ptr[1],
        //             frag_b_reg_ptr[2],
        //             frag_b_reg_ptr[3],
        //             frag_b_reg_ptr[4],
        //             frag_b_reg_ptr[5],
        //             frag_b_reg_ptr[6],
        //             frag_b_reg_ptr[7]
        //             );
        // }
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);

        int smem_write_stage_idx = Base::kStages - 1;
        int smem_read_stage_idx  = 0;

        //
        // Mainloop
        //

        __syncthreads();
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {
            //
            // Loop over GEMM K dimension
            //

            // Computes a warp-level GEMM on data held in shared memory
            // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

                // Load warp-level tiles from shared memory, wrapping to k offset if
                // this is the last group as the case may be.
                this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                ++this->warp_tile_iterator_A_;
                // static_assert(Base::kNumKIterationsPerWarpBLoad==1,"Base::kNumKIterationsPerWarpBLoad!=1");
                // static_assert(Base::kWarpGemmIterationsForB==4,"Base::kWarpGemmIterationsForB!=4");
                const int warp_tileB_k_compute_offset = warp_mma_k % Base::kNumKIterationsPerWarpBLoad;
                const int warp_tileB_k_load_offset    = warp_mma_k / Base::kNumKIterationsPerWarpBLoad;
                if (warp_tileB_k_compute_offset == Base::kNumKIterationsPerWarpBLoad - 1) {
                    this->warp_tile_iterator_B_.set_kgroup_index((warp_tileB_k_load_offset + 1)
                                                                 % Base::kWarpGemmIterationsForB);
                    this->warp_tile_iterator_B_.load(warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);
                    converted_frag_B_buffer[(warp_tileB_k_load_offset + 1) % 2] =
                        lds_converter(warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);

                    ++this->warp_tile_iterator_B_;
                    // if((threadIdx.x||threadIdx.y||threadIdx.z)==0){
                    //     uint32_t* frag_b_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);
                    //     printf("#### warp_frag_b load [%d] bid:%d-%d-%d,"
                    //                " frag_b_reg_ptr:%x-%x-%x-%x-%x-%x-%x-%x  \n",
                    //             ((warp_tileB_k_load_offset + 1) % 2),
                    //             blockIdx.x,blockIdx.y,blockIdx.z,
                    //             frag_b_reg_ptr[0],
                    //             frag_b_reg_ptr[1],
                    //             frag_b_reg_ptr[2],
                    //             frag_b_reg_ptr[3],
                    //             frag_b_reg_ptr[4],
                    //             frag_b_reg_ptr[5],
                    //             frag_b_reg_ptr[6],
                    //             frag_b_reg_ptr[7]
                    //             );
                    // }
                }
                // TODO(wangbojun) lds_converter can be remove for int8 B input
                // int4
                // typename TransformBAfterLDS::result_type converted_frag_B =
                //     lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2]);

                // typename TransformBAfterLDS::result_type converted_frag_B =
                //     lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2], reinterpret_cast<int32_t*>(nf4_smem_look_up_table));

                // typename TransformBAfterLDS::result_type converted_frag_B =
                //     lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2], reg_look_up_table);

                // typename TransformBAfterLDS::result_type converted_frag_B =
                //     lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2], shared_look_up_table, warp_idx, lane_idx);

                // if((threadIdx.x||threadIdx.y||threadIdx.z)==0){
                //     uint32_t* frag_b_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_B[(warp_tileB_k_load_offset) % 2]);
                //     uint32_t* converted_frag_B_reg_ptr = reinterpret_cast<uint32_t*>(&converted_frag_B);
                //     printf("#### after lds_converter bid:%d-%d-%d"
                //             " frag_b_reg_ptr[%d]:%x-%x-%x-%x-%x-%x-%x-%x"
                //             " converted_frag_b_reg_ptr:%x-%x-%x-%x-%x-%x-%x-%x  \n",
                //             blockIdx.x,blockIdx.y,blockIdx.z,
                //             ((warp_tileB_k_load_offset) % 2),
                //             frag_b_reg_ptr[0],
                //             frag_b_reg_ptr[1],
                //             frag_b_reg_ptr[2],
                //             frag_b_reg_ptr[3],
                //             frag_b_reg_ptr[4],
                //             frag_b_reg_ptr[5],
                //             frag_b_reg_ptr[6],
                //             frag_b_reg_ptr[7],
                //             converted_frag_B_reg_ptr[0],
                //             converted_frag_B_reg_ptr[1],
                //             converted_frag_B_reg_ptr[2],
                //             converted_frag_B_reg_ptr[3],
                //             converted_frag_B_reg_ptr[4],
                //             converted_frag_B_reg_ptr[5],
                //             converted_frag_B_reg_ptr[6],
                //             converted_frag_B_reg_ptr[7]
                //     );
                // }

                // bool  ::print_type< ::cutlass::Array< ::cutlass::integer_subbyte<(int)4, (bool)0> , (int)64, (bool)0> > ()")
                // print_type<WarpFragmentB>();
                // bool  ::print_type< ::cutlass::Array<signed char, (int)64, (bool)0> > ()") from a
                // print_type<TransformBAfterLDS::result_type>();
                // cutlass::Array<signed char, (int)32, (bool)0>
                // print_type<WarpFragmentA>();

                // print_type<FragmentC>();
                // TODO(zhengzekang)
                // run_warp_mma(


                // if(true){
                //     uint32_t none_zero = 0;
                //     // uint32_t* converted_frag_B_reg_ptr = reinterpret_cast<uint32_t*>(&converted_frag_B);
                //     uint32_t* converted_frag_B_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_B[warp_mma_k % 2]);
                //     uint32_t* frag_a_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_A[warp_mma_k % 2]);
                //     CUTLASS_PRAGMA_UNROLL
                //     for(int ii=0;ii<warp_frag_B[warp_mma_k % 2].size()/8;++ii){
                //         none_zero|=*(converted_frag_B_reg_ptr+ii);
                //     }
                //     CUTLASS_PRAGMA_UNROLL
                //     for(int ii=0;ii<warp_frag_A[warp_mma_k % 2].size()/4;++ii){
                //         none_zero|=*(frag_a_reg_ptr+ii);
                //     }

                //     CUTLASS_PRAGMA_UNROLL
                //     for(int none_zero_i = 16; none_zero_i>0;none_zero_i/=2){
                //         none_zero|= __shfl_xor_sync(-1,none_zero,none_zero_i);
                //     }

                //     if(none_zero!=0){
                //         printf("## before mma ## bidtid:%d-%d-%d-%d-%d-%d, warp_mma_k:%d, frag_B_reg_ptr:%x-%x-%x-%x-%x-%x-%x-%x; frag_a_reg_ptr:%x-%x-%x-%x-%x-%x-%x-%x"
                //                 " accu: %d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d \n",
                //                 blockIdx.x,blockIdx.y,blockIdx.z,
                //                 warp_mma_k,
                //                 threadIdx.x,threadIdx.y,threadIdx.z,
                //                 converted_frag_B_reg_ptr[0],
                //                 converted_frag_B_reg_ptr[1],
                //                 converted_frag_B_reg_ptr[2],
                //                 converted_frag_B_reg_ptr[3],
                //                 converted_frag_B_reg_ptr[4],
                //                 converted_frag_B_reg_ptr[5],
                //                 converted_frag_B_reg_ptr[6],
                //                 converted_frag_B_reg_ptr[7],
                //                 frag_a_reg_ptr[0],
                //                 frag_a_reg_ptr[1],
                //                 frag_a_reg_ptr[2],
                //                 frag_a_reg_ptr[3],
                //                 frag_a_reg_ptr[4],
                //                 frag_a_reg_ptr[5],
                //                 frag_a_reg_ptr[6],
                //                 frag_a_reg_ptr[7],
                //                 accum[0],
                //                 accum[1],
                //                 accum[2],
                //                 accum[3],
                //                 accum[4],
                //                 accum[5],
                //                 accum[6],
                //                 accum[7],
                //                 accum[8],
                //                 accum[9],
                //                 accum[10],
                //                 accum[11],
                //                 accum[12],
                //                 accum[13],
                //                 accum[14],
                //                 accum[15],
                //                 accum[16],
                //                 accum[17],
                //                 accum[18],
                //                 accum[19],
                //                 accum[20],
                //                 accum[21],
                //                 accum[22],
                //                 accum[23],
                //                 accum[24],
                //                 accum[25],
                //                 accum[26],
                //                 accum[27],
                //                 accum[28],
                //                 accum[29],
                //                 accum[30],
                //                 accum[31]
                //                );
                //     }
                // }
                run_ampere_warp_mma(
                    warp_mma, accum, warp_frag_A[warp_mma_k % 2], converted_frag_B_buffer[warp_tileB_k_load_offset % 2], accum, warp_tileB_k_compute_offset);
                // auto tmp = static_cast<int32_t>(warp_frag_B[warp_tileB_k_load_offset % 2]);
                // if(threadIdx.x==0  && threadIdx.y==0  && threadIdx.z==0 &&
                //     blockIdx.x==0  && blockIdx.y==0  && blockIdx.z==0){
                //     printf("### run_warp_mma: "
                //         "%d \n",
                //         reinterpret_cast<int32_t &>(accum));
                // }
                // if(true){
                //     uint32_t none_zero = 0;
                //     uint32_t* converted_frag_B_reg_ptr = reinterpret_cast<uint32_t*>(&converted_frag_B);
                //     // uint32_t* converted_frag_B_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_B[warp_mma_k % 2]);
                //     uint32_t* frag_a_reg_ptr = reinterpret_cast<uint32_t*>(&warp_frag_A[warp_mma_k % 2]);
                //     CUTLASS_PRAGMA_UNROLL
                //     for(int ii=0;ii<warp_frag_B[warp_mma_k % 2].size()/8;++ii){
                //         none_zero|=*(converted_frag_B_reg_ptr+ii);
                //     }
                //     CUTLASS_PRAGMA_UNROLL
                //     for(int ii=0;ii<warp_frag_A[warp_mma_k % 2].size()/4;++ii){
                //         none_zero|=*(frag_a_reg_ptr+ii);
                //     }
                //     CUTLASS_PRAGMA_UNROLL
                //     for(int none_zero_i = 16; none_zero_i>0;none_zero_i/=2){
                //         none_zero|= __shfl_xor_sync(-1,none_zero,none_zero_i);
                //     }

                //     // if(none_zero!=0){
                //     if((blockIdx.y||blockIdx.z||threadIdx.x||threadIdx.y||threadIdx.z)==0){

                //         printf("## after mma ## bidtid:%d-%d-%d-%d-%d-%d, warp_mma_k:%d, gemm_k_iterations:%d, Base::kWarpGemmIterations:%d,"
                //                 " converted_frag_B_reg_ptr:%x; frag_a_reg_ptr:%x"
                //                 " accu: %d \n",
                //                 blockIdx.x,blockIdx.y,blockIdx.z,
                //                 threadIdx.x,threadIdx.y,threadIdx.z,
                //                 warp_mma_k,
                //                 gemm_k_iterations,
                //                 Base::kWarpGemmIterations,
                //                 converted_frag_B_reg_ptr[0],
                //                 frag_a_reg_ptr[0],
                //                 accum[0]
                //                );
                //     }
                // }
                // Issue global->shared copies for the this stage
                if (warp_mma_k < Base::kWarpGemmIterations - 1) {
                    int group_start_iteration_A, group_start_iteration_B;

                    group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
                    group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

                    copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A, group_start_iteration_B);
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
                    int group_start_iteration_A, group_start_iteration_B;
                    group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
                    group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

                    copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A, group_start_iteration_B);

                    // Inserts a memory fence between stages of cp.async instructions.
                    cutlass::arch::cp_async_fence();

                    // Waits until kStages-2 stages have committed.
                    arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    // Move to the next stage
                    iterator_A.add_tile_offset({0, 1});
                    iterator_B.add_tile_offset({1, 0});

                    this->smem_iterator_A_.add_tile_offset({0, 1});
                    this->smem_iterator_B_.add_tile_offset({1, 0});

                    // Add negative offsets to return iterators to the 'start' of the
                    // circular buffer in shared memory
                    if (smem_write_stage_idx == (Base::kStages - 1)) {
                        this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
                        this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
                        smem_write_stage_idx = 0;
                    }
                    else {
                        ++smem_write_stage_idx;
                    }

                    if (smem_read_stage_idx == (Base::kStages - 1)) {
                        this->warp_tile_iterator_A_.add_tile_offset(
                            {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
                        this->warp_tile_iterator_B_.add_tile_offset(
                            {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterationsForB, 0});
                        smem_read_stage_idx = 0;
                    }
                    else {
                        ++smem_read_stage_idx;
                    }

                    --gemm_k_iterations;
                    iterator_A.clear_mask(gemm_k_iterations == 0);
                    iterator_B.clear_mask(gemm_k_iterations == 0);
                }
            }
        }

        if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            // commit and drain all pending and predicated LDGSTS pnz from the GEMM mainloop
            cutlass::arch::cp_async_fence();
            cutlass::arch::cp_async_wait<0>();
            __syncthreads();
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
