/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations
  targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

#include "cutlass_extensions/interleaved_numeric_conversion.h"

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of Scale elements
    typename ElementOperand_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    typename Enable = void>
class MmaTensorOpWin2xDequantizer {
    //static_assert(false, "Not Supported!");
};

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    /// Data type of Scale elements
    typename ElementOperand_>
class MmaTensorOpWin2xDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    ElementOperand_,
    layout::RowMajor,
    32>
    //typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 80
    //    && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
{
public:
    static_assert(platform::is_same<ElementOperand_, half_t>::value || platform::is_same<ElementOperand_, bfloat16_t>::value,
        "T must be fp16 or bf16");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of mma operand
    using ElementOperand = ElementOperand_;

    /// Type of input
    using ElementB = typename MmaOperator::FragmentB::Element;
    static_assert(platform::is_same<ElementB, uint2b_t>::value, "ElementB must be uint2b_t");

    /// Type of internal compute
    using ElementCompute = float;

    /// Type of the scales
    using ElementLocalScale = uint4b_t;
    using ElementSuperScale = ElementOperand;
    using ElementCodeScaleZp = float;

    /// Fragment to hold B data before Mma
    using FragmentInput = Array<ElementB, MmaOperator::FragmentB::kElements>;

    /// Unpack 4 uint2b_t values compreseed in a uint8_t to floating points.
    using Uint2Converter = FastInterleavedAndBiasedNumericArrayConverter<
        ElementOperand, ElementB, MmaOperator::FragmentB::kElements>;
    using FragmentUnpack = typename Uint2Converter::result_type;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    static constexpr int kElements = kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn;

    // 32 bits are loaded to register from shared memory by each thread
    static constexpr int kMmaIterationsPerLoad =
        32 / (sizeof_bits<ElementB>::value * ArchMmaOperator::FragmentB::kElements);

    // use uint8_t to save 2 4-bits local scales
    using FragmentLocalScale = Array<uint8_t, kElements>;
    using FragmentSuperScale = Array<ElementSuperScale, kElements>;
    using FragmentCodeScaleZp = Array<ElementCodeScaleZp, kElements>;

    /// Fragment to hold internal scales before Mma
    using FragmentCompute = Array<ElementCompute, kElements>;

    /// Fragment of dequantized B
    //using FragmentOutput = Array<ElementOperand, ArchMmaOperator::FragmentB::kElements * kElements>;
    using FragmentOutput = Array<ElementOperand, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using SuperTensorRef = cutlass::TensorRef<ElementSuperScale, Layout>;
    using LocalTensorRef = cutlass::TensorRef<ElementLocalScale, Layout>;
    using CodeTensorRef = cutlass::TensorRef<ElementCodeScaleZp, Layout>;

private:
    //
    // Data members
    //

    uint8_t* pointer_local_scale_;
    ElementCodeScaleZp* pointer_code_scale_;
    ElementCodeScaleZp* pointer_code_zp_;
    ElementSuperScale* pointer_super_scale_;

    FragmentUnpack unpacked_frag_;

public:
    CUTLASS_DEVICE
    MmaTensorOpWin2xDequantizer(SuperTensorRef smem_super_scale,
                                LocalTensorRef smem_local_scale,
                                CodeTensorRef smem_code_scale,
                                CodeTensorRef smem_code_zp,
                                int warp_idx_n,
                                int lane_idx) {
        int warp_offset = warp_idx_n * Shape::kN;
        int quad = lane_idx / 4;
        int thread_offset = warp_offset + quad;
        pointer_super_scale_ = smem_super_scale.data() + thread_offset;
        pointer_code_scale_ = smem_code_scale.data() + thread_offset;
        pointer_code_zp_ = smem_code_zp.data() + thread_offset;
        pointer_local_scale_ = reinterpret_cast<uint8_t *>(smem_local_scale.data()) + thread_offset;
    }

    /// Channel-wise params, need to load just once
    CUTLASS_DEVICE
    void load(FragmentCodeScaleZp& code_scale_frag,
              FragmentCodeScaleZp& code_zp_frag,
              FragmentSuperScale& super_scale_frag) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            super_scale_frag[mma_n_iter] = pointer_super_scale_[mma_n_iter * InstructionShape::kN]; // bank conflict
            code_scale_frag[mma_n_iter] = pointer_code_scale_[mma_n_iter * InstructionShape::kN];
            code_zp_frag[mma_n_iter] = pointer_code_zp_[mma_n_iter * InstructionShape::kN];
        }
    }

    /// Group-wise params, need to load multiple times
    CUTLASS_DEVICE
    void load(FragmentLocalScale& local_scale_frag) {
        //CUTLASS_TRACE_DEVICE(" pointer_local_scale_=%p", pointer_local_scale_);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
            local_scale_frag[mma_n_iter] = pointer_local_scale_[mma_n_iter * InstructionShape::kN]; // bank conflict
        }
    }

    CUTLASS_DEVICE
    void dequantize(const FragmentLocalScale& local_scale_frag,
                    const FragmentCodeScaleZp& code_scale_frag,
                    const FragmentCodeScaleZp& code_zp_frag,
                    const FragmentSuperScale& super_scale_frag,
                    const FragmentInput& input_frag,
                    FragmentOutput& output_frag,
                    int tb_offset_k) {
        int stage = tb_offset_k / 64;

        //CUTLASS_TRACE_DEVICE(" FragmentInput::kElements=%d, %d bytes",
        //    FragmentInput::kElements, static_cast<int>(sizeof_bits<FragmentInput>::value / 8));
        //CUTLASS_TRACE_DEVICE(" FragmentUnpack::kElements=%d, %d bytes",
        //    FragmentUnpack::kElements, static_cast<int>(sizeof_bits<FragmentUnpack>::value / 8));
        //CUTLASS_TRACE_DEVICE(" FragmentOutput::kElements=%d, %d bytes",
        //    FragmentOutput::kElements, static_cast<int>(sizeof_bits<FragmentOutput>::value / 8));

        //CUTLASS_TRACE_DEVICE(" MmaOperator::FragmentB::kElements=%d", MmaOperator::FragmentB::kElements);
        //CUTLASS_TRACE_DEVICE(" MmaOperator::IteratorB::InstructionShape: %dx%d; InstructionShape: %dx%dx%d; ",
        //    MmaOperator::IteratorB::InstructionShape::kRow, MmaOperator::IteratorB::InstructionShape::kColumn,
        //    InstructionShape::kM, InstructionShape::kN, InstructionShape::kK);
        //CUTLASS_TRACE_DEVICE(" MmaOperator::MmaIterations: kRow=%d, kColumn=%d",
        //    MmaOperator::MmaIterations::kRow, MmaOperator::MmaIterations::kColumn);

        unpacked_frag_ = Uint2Converter::convert(input_frag);
        // DEBUG CODES
        for (int i = 0; i < FragmentUnpack::kElements; ++i) {
            unpacked_frag_[i] = static_cast<typename FragmentUnpack::Element>(1); //static_cast<typename FragmentUnpack::Element>((i / 16) * 8 + (threadIdx.x % 32) / 4);
        }

#if 0
        if (FragmentUnpack::kElements == 64) {
            CUTLASS_TRACE_DEVICE(" unpacked_frag_[0:15]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                static_cast<float>(unpacked_frag_[0]), static_cast<float>(unpacked_frag_[1]),
                static_cast<float>(unpacked_frag_[2]), static_cast<float>(unpacked_frag_[3]),
                static_cast<float>(unpacked_frag_[4]), static_cast<float>(unpacked_frag_[5]),
                static_cast<float>(unpacked_frag_[6]), static_cast<float>(unpacked_frag_[7]),
                static_cast<float>(unpacked_frag_[8]), static_cast<float>(unpacked_frag_[9]),
                static_cast<float>(unpacked_frag_[10]), static_cast<float>(unpacked_frag_[11]),
                static_cast<float>(unpacked_frag_[12]), static_cast<float>(unpacked_frag_[13]),
                static_cast<float>(unpacked_frag_[14]), static_cast<float>(unpacked_frag_[15]));
            CUTLASS_TRACE_DEVICE(" unpacked_frag_[16:31]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                static_cast<float>(unpacked_frag_[16]), static_cast<float>(unpacked_frag_[17]),
                static_cast<float>(unpacked_frag_[18]), static_cast<float>(unpacked_frag_[19]),
                static_cast<float>(unpacked_frag_[20]), static_cast<float>(unpacked_frag_[21]),
                static_cast<float>(unpacked_frag_[22]), static_cast<float>(unpacked_frag_[23]),
                static_cast<float>(unpacked_frag_[24]), static_cast<float>(unpacked_frag_[25]),
                static_cast<float>(unpacked_frag_[26]), static_cast<float>(unpacked_frag_[27]),
                static_cast<float>(unpacked_frag_[28]), static_cast<float>(unpacked_frag_[29]),
                static_cast<float>(unpacked_frag_[30]), static_cast<float>(unpacked_frag_[31]));
        }
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

        static constexpr int32_t kGroupSize = 64;
        static constexpr int32_t kLocalScaleMask = 0xF;

        // special for TileRows = 64
        int local_scale_shift = (((tb_offset_k / kGroupSize) + 1) & 1) * 4;
        FragmentCompute scale_frag;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentLocalScale::kElements; ++i) {
            int32_t shifted_local_scale =
                (static_cast<int32_t>(local_scale_frag[i]) >> local_scale_shift) & kLocalScaleMask;
            scale_frag[i] =
                static_cast<ElementCompute>(shifted_local_scale) * static_cast<ElementCompute>(super_scale_frag[i]);
        }

#if 0
        if (FragmentCompute::kElements == 4) {
        CUTLASS_TRACE_DEVICE(" [stage=%d] tb_offset_k=%d, local_scale_shift=%d, scale_frag[0:3]=[%f, %f, %f, %f], sizeof(FragmentCompute)=%d bytes",
                stage, tb_offset_k, local_scale_shift,
                static_cast<float>(scale_frag[0]), static_cast<float>(scale_frag[1]),
                static_cast<float>(scale_frag[2]), static_cast<float>(scale_frag[3]),
                static_cast<int>(sizeof(FragmentCompute)));
        }
#endif

        //int offset = warp_mma_k * ArchMmaOperator::FragmentB::kElements;
        int num_columns = 32 / sizeof_bits<ElementB>::value;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < num_columns; ++j) {
                ElementCompute scaled_value =
                    static_cast<ElementCompute>(unpacked_frag_[mma_n_iter * num_columns + j]) * scale_frag[mma_n_iter];
                output_frag[mma_n_iter * num_columns + j] = static_cast<ElementOperand>(scaled_value);
            }
        }

        if (FragmentOutput::kElements == 64) {
#if 0
            CUTLASS_TRACE_DEVICE(" [stage=%d] output_frag[0:15]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                stage,
                static_cast<float>(output_frag[0]), static_cast<float>(output_frag[1]),
                static_cast<float>(output_frag[2]), static_cast<float>(output_frag[3]),
                static_cast<float>(output_frag[4]), static_cast<float>(output_frag[5]),
                static_cast<float>(output_frag[6]), static_cast<float>(output_frag[7]),
                static_cast<float>(output_frag[8]), static_cast<float>(output_frag[9]),
                static_cast<float>(output_frag[10]), static_cast<float>(output_frag[11]),
                static_cast<float>(output_frag[12]), static_cast<float>(output_frag[13]),
                static_cast<float>(output_frag[14]), static_cast<float>(output_frag[15]));
            CUTLASS_TRACE_DEVICE(" [stage=%d] output_frag[16:31]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                stage,
                static_cast<float>(output_frag[16]), static_cast<float>(output_frag[17]),
                static_cast<float>(output_frag[18]), static_cast<float>(output_frag[19]),
                static_cast<float>(output_frag[20]), static_cast<float>(output_frag[21]),
                static_cast<float>(output_frag[22]), static_cast<float>(output_frag[23]),
                static_cast<float>(output_frag[24]), static_cast<float>(output_frag[25]),
                static_cast<float>(output_frag[26]), static_cast<float>(output_frag[27]),
                static_cast<float>(output_frag[28]), static_cast<float>(output_frag[29]),
                static_cast<float>(output_frag[30]), static_cast<float>(output_frag[31]));
            CUTLASS_TRACE_DEVICE(" [stage=%d] output_frag[32:47]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                stage,
                static_cast<float>(output_frag[32]), static_cast<float>(output_frag[33]),
                static_cast<float>(output_frag[34]), static_cast<float>(output_frag[35]),
                static_cast<float>(output_frag[36]), static_cast<float>(output_frag[37]),
                static_cast<float>(output_frag[38]), static_cast<float>(output_frag[39]),
                static_cast<float>(output_frag[40]), static_cast<float>(output_frag[41]),
                static_cast<float>(output_frag[42]), static_cast<float>(output_frag[43]),
                static_cast<float>(output_frag[44]), static_cast<float>(output_frag[45]),
                static_cast<float>(output_frag[46]), static_cast<float>(output_frag[47]));
            CUTLASS_TRACE_DEVICE(" [stage=%d] output_frag[48:63]=[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]",
                stage,
                static_cast<float>(output_frag[48]), static_cast<float>(output_frag[49]),
                static_cast<float>(output_frag[50]), static_cast<float>(output_frag[51]),
                static_cast<float>(output_frag[52]), static_cast<float>(output_frag[53]),
                static_cast<float>(output_frag[54]), static_cast<float>(output_frag[55]),
                static_cast<float>(output_frag[56]), static_cast<float>(output_frag[57]),
                static_cast<float>(output_frag[58]), static_cast<float>(output_frag[59]),
                static_cast<float>(output_frag[60]), static_cast<float>(output_frag[61]),
                static_cast<float>(output_frag[62]), static_cast<float>(output_frag[63]));
#endif
        }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on
        // older arch, scale conversion should happen before scales are stored
        // to shared memory and we should use the fp16 dequantizer. This will
        // avoid numerous conversion instructions in GEMM main loop.
        arch::device_breakpoint();
#endif

        const int fixed_values[64] = {
            0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
            2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59,
            4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61,
            6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63
        };
        for (int i = 0; i < FragmentUnpack::kElements; ++i) {
            output_frag[i] = static_cast<typename FragmentUnpack::Element>(fixed_values[(i % 16) + (threadIdx.x % 4) * 16]);
        }
    }

    /// Add an offset to pointer in units of elements.
    /// Only group-wise params needs.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset) {
        pointer_local_scale_ += offset;
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
