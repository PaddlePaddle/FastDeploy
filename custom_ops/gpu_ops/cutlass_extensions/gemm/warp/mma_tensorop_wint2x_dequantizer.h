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

#include <cuda_bf16.h>
#include "cutlass_extensions/weight_only_quant_op.h"

////////////////////////////////////////////////////////////////////////////////

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
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    WeightOnlyQuantOp QuantOp_ = WeightOnlyQuantOp::UNDEFINED,
    ///
    typename Enable = void>
class MmaTensorOpWin2xDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>

class MmaTensorOpWin2xDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    bfloat16_t,
    layout::ColumnMajor,
    32,
    QuantOp_,
    typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >=
                                 70>::type> {
   public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor =
        MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementWeight = uint2b_t;

    /// Type of the scales
    using ElementUnzipWeight = uint8_t;

    /// Type of the scales
    using ElementScale = bfloat16_t;

    /// Type of the scales
    using ScaleComputeT = float;

    static constexpr int unzip_len = 4;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand =
        Array<ElementWeight, MmaOperator::FragmentB::kElements>;
    using FragmentWeightOperand =
        Array<ElementUnzipWeight,
              MmaOperator::FragmentB::kElements / unzip_len>;
    using FragmentOutOperand =
        Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentLocalScale = Array<ElementScale, 1>;
    using FragmentCodeScale = Array<ScaleComputeT, 1>;
    using FragmentCodeZp = Array<ScaleComputeT, 1>;
    using FragmentSuperScale = Array<ElementScale, 1>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::ColumnMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = cutlass::TensorRef<ElementScale, Layout>;
    using TensorCodeRef = cutlass::TensorRef<ScaleComputeT, Layout>;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

    CUTLASS_DEVICE
    MmaTensorOpWin2xDequantizer(TensorRef smem_local_scale,
                                TensorCodeRef smem_code_scale,
                                TensorCodeRef smem_code_zp,
                                TensorRef smem_super_scale,
                                int const warp_idx_n,
                                int const lane_idx) {
        int const warp_offset = warp_idx_n * Shape::kN;
        int const quad = lane_idx / 4;
        int const thread_offset = warp_offset + quad;
        pointer_local_scale_ = smem_local_scale.data() + thread_offset;
        pointer_code_scale_ = smem_code_scale.data() + thread_offset;
        pointer_code_zp_ = smem_code_zp.data() + thread_offset;
        if constexpr (hasZero(QuantOp)) {
            pointer_super_scale_ = smem_super_scale.data() + thread_offset;
        }
    }

    // CUTLASS_DEVICE
    // MmaTensorOpWin2xDequantizer() {
    //     pointer_local_scale_ = nullptr;
    //     pointer_code_scale_ = nullptr;
    //     pointer_code_zp_ = nullptr;
    //     if constexpr (hasZero(QuantOp)) {
    //         pointer_super_scale_ = nullptr;
    //     }
    // }

    CUTLASS_DEVICE
    MmaTensorOpWin2xDequantizer() {
        // Create fake pointer using a shared dummy buffer
        CUTLASS_TRACE_DEVICE(" warp dequant aaa");

        extern __shared__ char cutlass_fake_dequant_smem[];

        // Memory layout (manual alignment):
        // ElementScale (half or bf16): 2 bytes
        // ScaleComputeT (float): 4 bytes

        pointer_local_scale_ =
            reinterpret_cast<ElementScale*>(cutlass_fake_dequant_smem);
        pointer_code_scale_ =
            reinterpret_cast<ScaleComputeT*>(cutlass_fake_dequant_smem + 64);
        pointer_code_zp_ =
            reinterpret_cast<ScaleComputeT*>(cutlass_fake_dequant_smem + 128);

        if constexpr (hasZero(QuantOp)) {
            pointer_super_scale_ = reinterpret_cast<ElementScale*>(
                cutlass_fake_dequant_smem + 192);
        }
    }

    CUTLASS_DEVICE
    void load(FragmentLocalScale& local_scale_frag,
              FragmentCodeScale& code_scale_frag,
              FragmentCodeZp& code_zp_frag,
              FragmentSuperScale& super_scale_frag) {
        CUTLASS_TRACE_DEVICE(" warp dequant load");
        // CUTLASS_PRAGMA_UNROLL
        // for (int mma_n_iter = 0; mma_n_iter <
        // MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        // {
        //     local_scale_frag[mma_n_iter] = pointer_local_scale_[mma_n_iter *
        //     InstructionShape::kN]; code_scale_frag[mma_n_iter] =
        //     pointer_code_scale_[mma_n_iter * InstructionShape::kN];
        //     code_zp_frag[mma_n_iter] = pointer_code_zp_[mma_n_iter *
        //     InstructionShape::kN]; if constexpr (hasZero(QuantOp))
        //     {
        //         super_scale_frag[mma_n_iter] =
        //         pointer_super_scale_[mma_n_iter * InstructionShape::kN];
        //     }
        // }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentOutOperand& out_frag,
                    FragmentDequantizedOperand& operand_frag,
                    FragmentLocalScale& local_scale_frag,
                    FragmentCodeScale& code_scale_frag,
                    FragmentCodeZp& code_zp_frag,
                    FragmentSuperScale& super_scale_frag) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        CUTLASS_TRACE_DEVICE(" dequantize if def");

        static constexpr int32_t kGroupSize = 64;
        static constexpr int32_t kPackNum = 4;
        static constexpr int32_t kWeightMask = 0x3F;
        static constexpr int32_t kLocalScaleMask = 0xF;
        static constexpr int32_t kBZP = 32;

        // using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        // using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element,
        // kExpansionFactor * _MmaOperandB::kElements>;
        // static_assert(ExpandedMmaOperandB::kElements *
        // MmaOperator::MmaIterations::kColumn
        //         == FragmentDequantizedOperand::kElements,
        //     "");

        // CUTLASS_TRACE_DEVICE(" MmaIterations krow = %d, kcol = %d",
        // MmaOperator::IteratorB::InstructionShape::kRow,
        // MmaOperator::MmaIterations::kColumn);

        // CUTLASS_TRACE_DEVICE(" kExpansionFactor = %d / %d",
        // MmaOperator::IteratorB::InstructionShape::kRow,
        // InstructionShape::kK); CUTLASS_TRACE_DEVICE("
        // FragmentDequantizedOperand::kElements = %d ",
        // FragmentDequantizedOperand::kElements); CUTLASS_TRACE_DEVICE("
        // _MmaOperandB::kElements = %d ",  _MmaOperandB::kElements);

        // FragmentWeightOperand
        CUTLASS_TRACE_DEVICE(" FragmentWeightOperand elem = %d ",
                             FragmentWeightOperand::kElements);
        // CUTLASS_TRACE_DEVICE(" ElementUnzipWeight size = %d ",
        // sizeof(ElementUnzipWeight)); CUTLASS_TRACE_DEVICE(" ElementWeight
        // size = %d ",  sizeof(ElementWeight));
        static_assert(std::is_same<typename FragmentWeightOperand::Element,
                                   cutlass::uint8_t>::value,
                      "B 是 uint8 量化类型");
        FragmentWeightOperand* weight_ptr =
            reinterpret_cast<FragmentWeightOperand*>(&operand_frag);
        FragmentLocalScale* local_scale_ptr =
            reinterpret_cast<FragmentLocalScale*>(&local_scale_frag);
        FragmentCodeScale* code_scale_ptr =
            reinterpret_cast<FragmentCodeScale*>(&code_scale_frag);
        FragmentCodeZp* code_zp_ptr =
            reinterpret_cast<FragmentCodeZp*>(&code_zp_frag);
        FragmentSuperScale* super_scale_ptr =
            reinterpret_cast<FragmentSuperScale*>(&super_scale_frag);

        ScaleComputeT code_scale =
            static_cast<ScaleComputeT>(code_scale_ptr[0][0]);
        ScaleComputeT code_zp = static_cast<ScaleComputeT>(code_zp_ptr[0][0]);
        ScaleComputeT super_scale =
            static_cast<ScaleComputeT>(super_scale_ptr[0][0]);
        int32_t local_scale = static_cast<int32_t>(local_scale_ptr[0][0]);
        int32_t const shift_bits[4] = {9, 6, 3, 0};

        ScaleComputeT zipped_value[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            zipped_value[i] = static_cast<ScaleComputeT>(weight_ptr[0][i]);
        }

        int local_scale_shift = 4;
        int32_t shifted_local_scale =
            (local_scale >> local_scale_shift) & kLocalScaleMask;
        ScaleComputeT scale =
            static_cast<ScaleComputeT>(shifted_local_scale) * super_scale;

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int32_t decode_value = static_cast<int32_t>(
                floor(zipped_value[i] * code_scale + code_zp +
                      static_cast<ScaleComputeT>(0.5)));

            int col = i * 4;

#pragma unroll
            for (int shift_bit_id = 0; shift_bit_id < 4; ++shift_bit_id) {
                int32_t shift_bit = shift_bits[shift_bit_id];
                int32_t shifted_value =
                    (decode_value >> shift_bit) & kWeightMask;

                ScaleComputeT value =
                    static_cast<ScaleComputeT>(shifted_value - kBZP);
                out_frag[col + shift_bit_id] =
                    static_cast<ElementScale>(scale * value);
            }
        }

        CUTLASS_TRACE_DEVICE(" kColsPerMmaPerThread = %d ",
                             kColsPerMmaPerThread);
        CUTLASS_TRACE_DEVICE(" MmaOperator::MmaIterations::kColumn = %d ",
                             MmaOperator::MmaIterations::kColumn);

        // // __nv_bfloat16 const* scale_ptr = reinterpret_cast<__nv_bfloat16
        // const*>(&scale_frag); ExpandedMmaOperandB* operand_frag_ptr =
        // reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

        // printf("threadidx.x = %d\n", threadIdx.x);
        // CUTLASS_PRAGMA_UNROLL
        // for (int mma_n_iter = 0; mma_n_iter <
        // MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        // {
        //     static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

        //     __nv_bfloat162 scalex2 =
        //     __bfloat162bfloat162(scale_ptr[mma_n_iter]);
        //     __nv_bfloat162* operand_bf16x2_ptr =
        //     reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

        //     CUTLASS_PRAGMA_UNROLL
        //     for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
        //     {
        //         operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii],
        //         scalex2);
        //     }
        // }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on
        // older arch, scale conversion should happen before scales are stored
        // to shared memory and we should use the fp16 dequantizer. This will
        // avoid numerous conversion instructions in GEMM main loop.
        CUTLASS_TRACE_DEVICE(" dequantize else def");
        // arch::device_breakpoint();
#endif
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset) {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_local_scale_ += offset;
        pointer_code_scale_ += offset;
        pointer_code_zp_ += offset;
        pointer_super_scale_ += offset;
    }

   private:
    ElementScale const* pointer_local_scale_;
    ScaleComputeT const* pointer_code_scale_;
    ScaleComputeT const* pointer_code_zp_;
    ElementScale const* pointer_super_scale_;

    ElementScale const* pointer_out_;
};

template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpWin2xDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::ColumnMajor,
    32,
    QuantOp_,
    typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >=
                                 70>::type> {
   public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor =
        MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementWeight = uint2b_t;

    /// Type of the scales
    using ElementUnzipWeight = uint8_t;

    /// Type of the scales
    using ElementScale = half_t;

    /// Type of the scales
    using ScaleComputeT = float;

    static constexpr int unzip_len = 4;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand =
        Array<ElementWeight, MmaOperator::FragmentB::kElements>;
    using FragmentWeightOperand =
        Array<ElementUnzipWeight,
              MmaOperator::FragmentB::kElements / unzip_len>;
    using FragmentOutOperand =
        Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentLocalScale = Array<ElementScale, 1>;
    using FragmentCodeScale = Array<ScaleComputeT, 1>;
    using FragmentCodeZp = Array<ScaleComputeT, 1>;
    using FragmentSuperScale = Array<ElementScale, 1>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::ColumnMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = cutlass::TensorRef<ElementScale, Layout>;
    using TensorCodeRef = cutlass::TensorRef<ScaleComputeT, Layout>;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

    CUTLASS_DEVICE
    MmaTensorOpWin2xDequantizer(TensorRef smem_local_scale,
                                TensorCodeRef smem_code_scale,
                                TensorCodeRef smem_code_zp,
                                TensorRef smem_super_scale,
                                int const warp_idx_n,
                                int const lane_idx) {
        int const warp_offset = warp_idx_n * Shape::kN;
        int const quad = lane_idx / 4;
        int const thread_offset = warp_offset + quad;
        pointer_local_scale_ = smem_local_scale.data() + thread_offset;
        pointer_code_scale_ = smem_code_scale.data() + thread_offset;
        pointer_code_zp_ = smem_code_zp.data() + thread_offset;
        if constexpr (hasZero(QuantOp)) {
            pointer_super_scale_ = smem_super_scale.data() + thread_offset;
        }
    }

    // CUTLASS_DEVICE
    // MmaTensorOpWin2xDequantizer() {
    //     pointer_local_scale_ = nullptr;
    //     pointer_code_scale_ = nullptr;
    //     pointer_code_zp_ = nullptr;
    //     if constexpr (hasZero(QuantOp)) {
    //         pointer_super_scale_ = nullptr;
    //     }
    // }

    CUTLASS_DEVICE
    MmaTensorOpWin2xDequantizer() {
        // Create fake pointer using a shared dummy buffer
        CUTLASS_TRACE_DEVICE(" warp dequant aaa");

        extern __shared__ char cutlass_fake_dequant_smem[];

        // Memory layout (manual alignment):
        // ElementScale (half or bf16): 2 bytes
        // ScaleComputeT (float): 4 bytes

        pointer_local_scale_ =
            reinterpret_cast<ElementScale*>(cutlass_fake_dequant_smem);
        pointer_code_scale_ =
            reinterpret_cast<ScaleComputeT*>(cutlass_fake_dequant_smem + 64);
        pointer_code_zp_ =
            reinterpret_cast<ScaleComputeT*>(cutlass_fake_dequant_smem + 128);

        if constexpr (hasZero(QuantOp)) {
            pointer_super_scale_ = reinterpret_cast<ElementScale*>(
                cutlass_fake_dequant_smem + 192);
        }
    }

    CUTLASS_DEVICE
    void load(FragmentLocalScale& local_scale_frag,
              FragmentCodeScale& code_scale_frag,
              FragmentCodeZp& code_zp_frag,
              FragmentSuperScale& super_scale_frag) {
        CUTLASS_TRACE_DEVICE(" warp dequant load");
        // CUTLASS_PRAGMA_UNROLL
        // for (int mma_n_iter = 0; mma_n_iter <
        // MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        // {
        //     local_scale_frag[mma_n_iter] = pointer_local_scale_[mma_n_iter *
        //     InstructionShape::kN]; code_scale_frag[mma_n_iter] =
        //     pointer_code_scale_[mma_n_iter * InstructionShape::kN];
        //     code_zp_frag[mma_n_iter] = pointer_code_zp_[mma_n_iter *
        //     InstructionShape::kN]; if constexpr (hasZero(QuantOp))
        //     {
        //         super_scale_frag[mma_n_iter] =
        //         pointer_super_scale_[mma_n_iter * InstructionShape::kN];
        //     }
        // }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentOutOperand& out_frag,
                    FragmentDequantizedOperand& operand_frag,
                    FragmentLocalScale& local_scale_frag,
                    FragmentCodeScale& code_scale_frag,
                    FragmentCodeZp& code_zp_frag,
                    FragmentSuperScale& super_scale_frag) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        CUTLASS_TRACE_DEVICE(" dequantize if def");

        static constexpr int32_t kGroupSize = 64;
        static constexpr int32_t kPackNum = 4;
        static constexpr int32_t kWeightMask = 0x3F;
        static constexpr int32_t kLocalScaleMask = 0xF;
        static constexpr int32_t kBZP = 32;

        // using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        // using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element,
        // kExpansionFactor * _MmaOperandB::kElements>;
        // static_assert(ExpandedMmaOperandB::kElements *
        // MmaOperator::MmaIterations::kColumn
        //         == FragmentDequantizedOperand::kElements,
        //     "");

        // CUTLASS_TRACE_DEVICE(" MmaIterations krow = %d, kcol = %d",
        // MmaOperator::IteratorB::InstructionShape::kRow,
        // MmaOperator::MmaIterations::kColumn);

        // CUTLASS_TRACE_DEVICE(" kExpansionFactor = %d / %d",
        // MmaOperator::IteratorB::InstructionShape::kRow,
        // InstructionShape::kK); CUTLASS_TRACE_DEVICE("
        // FragmentDequantizedOperand::kElements = %d ",
        // FragmentDequantizedOperand::kElements); CUTLASS_TRACE_DEVICE("
        // _MmaOperandB::kElements = %d ",  _MmaOperandB::kElements);

        // FragmentWeightOperand
        CUTLASS_TRACE_DEVICE(" FragmentWeightOperand elem = %d ",
                             FragmentWeightOperand::kElements);
        // CUTLASS_TRACE_DEVICE(" ElementUnzipWeight size = %d ",
        // sizeof(ElementUnzipWeight)); CUTLASS_TRACE_DEVICE(" ElementWeight
        // size = %d ",  sizeof(ElementWeight));
        static_assert(std::is_same<typename FragmentWeightOperand::Element,
                                   cutlass::uint8_t>::value,
                      "B 是 uint8 量化类型");
        FragmentWeightOperand* weight_ptr =
            reinterpret_cast<FragmentWeightOperand*>(&operand_frag);
        FragmentLocalScale* local_scale_ptr =
            reinterpret_cast<FragmentLocalScale*>(&local_scale_frag);
        FragmentCodeScale* code_scale_ptr =
            reinterpret_cast<FragmentCodeScale*>(&code_scale_frag);
        FragmentCodeZp* code_zp_ptr =
            reinterpret_cast<FragmentCodeZp*>(&code_zp_frag);
        FragmentSuperScale* super_scale_ptr =
            reinterpret_cast<FragmentSuperScale*>(&super_scale_frag);

        ScaleComputeT code_scale =
            static_cast<ScaleComputeT>(code_scale_ptr[0][0]);
        ScaleComputeT code_zp = static_cast<ScaleComputeT>(code_zp_ptr[0][0]);
        ScaleComputeT super_scale =
            static_cast<ScaleComputeT>(super_scale_ptr[0][0]);
        int32_t local_scale = static_cast<int32_t>(local_scale_ptr[0][0]);
        int32_t const shift_bits[4] = {9, 6, 3, 0};

        ScaleComputeT zipped_value[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            zipped_value[i] = static_cast<ScaleComputeT>(weight_ptr[0][i]);
        }

        int local_scale_shift = 4;
        int32_t shifted_local_scale =
            (local_scale >> local_scale_shift) & kLocalScaleMask;
        ScaleComputeT scale =
            static_cast<ScaleComputeT>(shifted_local_scale) * super_scale;

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int32_t decode_value = static_cast<int32_t>(
                floor(zipped_value[i] * code_scale + code_zp +
                      static_cast<ScaleComputeT>(0.5)));

            int col = i * 4;

#pragma unroll
            for (int shift_bit_id = 0; shift_bit_id < 4; ++shift_bit_id) {
                int32_t shift_bit = shift_bits[shift_bit_id];
                int32_t shifted_value =
                    (decode_value >> shift_bit) & kWeightMask;

                ScaleComputeT value =
                    static_cast<ScaleComputeT>(shifted_value - kBZP);
                out_frag[col + shift_bit_id] =
                    static_cast<ElementScale>(scale * value);
            }
        }

        CUTLASS_TRACE_DEVICE(" kColsPerMmaPerThread = %d ",
                             kColsPerMmaPerThread);
        CUTLASS_TRACE_DEVICE(" MmaOperator::MmaIterations::kColumn = %d ",
                             MmaOperator::MmaIterations::kColumn);

        // // __nv_bfloat16 const* scale_ptr = reinterpret_cast<__nv_bfloat16
        // const*>(&scale_frag); ExpandedMmaOperandB* operand_frag_ptr =
        // reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

        // printf("threadidx.x = %d\n", threadIdx.x);
        // CUTLASS_PRAGMA_UNROLL
        // for (int mma_n_iter = 0; mma_n_iter <
        // MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        // {
        //     static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

        //     __nv_bfloat162 scalex2 =
        //     __bfloat162bfloat162(scale_ptr[mma_n_iter]);
        //     __nv_bfloat162* operand_bf16x2_ptr =
        //     reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

        //     CUTLASS_PRAGMA_UNROLL
        //     for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
        //     {
        //         operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii],
        //         scalex2);
        //     }
        // }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on
        // older arch, scale conversion should happen before scales are stored
        // to shared memory and we should use the fp16 dequantizer. This will
        // avoid numerous conversion instructions in GEMM main loop.
        CUTLASS_TRACE_DEVICE(" dequantize else def");
        // arch::device_breakpoint();
#endif
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset) {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_local_scale_ += offset;
        pointer_code_scale_ += offset;
        pointer_code_zp_ += offset;
        pointer_super_scale_ += offset;
    }

   private:
    ElementScale const* pointer_local_scale_;
    ScaleComputeT const* pointer_code_scale_;
    ScaleComputeT const* pointer_code_zp_;
    ElementScale const* pointer_super_scale_;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
