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

namespace detail {

template <typename T, int InstructSize>
struct Multiplier {
    using Element = T;
    using Fragment = Array<Element, InstructSize>;

    template <int N>
    CUTLASS_DEVICE
    static void Compute(Array<Element, N> const& operand_frag, Element scalar, Array<Element, N> &result_frag) {
        arch::device_breakpoint();
    }
};

template <>
struct Multiplier<bfloat16_t, 2> {
    using Element = bfloat16_t;
    using Fragment = Array<Element, 2>;

    template <int N>
    CUTLASS_DEVICE
    static void Compute(Array<Element, N> const& operand_frag, Element scalar, Array<Element, N> &result_frag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

        __nv_bfloat16 const* scalar_ptr = reinterpret_cast<__nv_bfloat16 const*>(&scalar);
        Fragment const* operand_ptr = reinterpret_cast<Fragment const*>(&operand_frag);
        Fragment* result_ptr = reinterpret_cast<Fragment *>(&result_frag);

        __nv_bfloat162 scalarx2 = __bfloat162bfloat162(*scalar_ptr);
        __nv_bfloat162 const* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162 const*>(&operand_ptr);
        __nv_bfloat162* result_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&result_ptr);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 2; ++i) {
            result_bf16x2_ptr[i] = __hmul2(operand_bf16x2_ptr[i], scalarx2);
        }

#else
        // Slow path not implemented here on purpose. If we need to do HMMA on
        // older arch, scale conversion should happen before scales are stored
        // to shared memory and we should use the fp16 dequantizer. This will
        // avoid numerous conversion instructions in GEMM main loop.
        arch::device_breakpoint();
#endif
    }
};

} // namespace detail

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
    /// Group size for quantization
    int GroupSize_,
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
    typename ElementOperand_,
    /// Group size for quantization
    int GroupSize_>
class MmaTensorOpWin2xDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    ElementOperand_,
    layout::RowMajor,
    GroupSize_>
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

    /// Warp mma shape
    using Shape = Shape_;

    /// Type of mma operand
    using ElementOperand = ElementOperand_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// Group size for quantization
    static constexpr int kGroupSize = GroupSize_;

    /// Type of input
    using ElementB = typename MmaOperator::FragmentB::Element;
    static_assert(platform::is_same<ElementB, uint2b_t>::value, "ElementB must be uint2b_t");

    /// Type of internal compute
    using ElementCompute = ElementOperand;

    /// Type of the scales
    using ElementLocalScale = uint4b_t;
    using ElementSuperScale = ElementOperand;
    using ElementCodeScaleZp = float;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kWarpIterationsAlongN = MmaOperator::MmaIterations::kColumn;

    // use uint8_t to save 2 4-bits local scales
    using FragmentLocalScale = Array<uint8_t, kWarpIterationsAlongN>;
    using FragmentSuperScale = Array<ElementSuperScale, kWarpIterationsAlongN>;
    using FragmentCodeScaleZp = Array<ElementCodeScaleZp, kWarpIterationsAlongN>;

    /// Fragment to hold B data before Mma
    using FragmentInput = Array<ElementB, MmaOperator::FragmentB::kElements>;

    /// Unpack 4 uint2b_t values compressed in a uint8_t to floating points
    using Uint2Converter = FastInterleavedAndBiasedNumericArrayConverter<
        ElementOperand, ElementB, MmaOperator::FragmentB::kElements>;
    using FragmentInputUnpack = typename Uint2Converter::result_type;

    /// Fragment to hold internal scales before Mma
    using FragmentScale = Array<ElementCompute, FragmentLocalScale::kElements>;

    /// This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Fragment of dequantized B
    using FragmentOutput = Array<ElementOperand, MmaOperator::FragmentB::kElements / kExpansionFactor>;

    //using Multiplier = detail::Multiplier<ElementOperand, 2>;

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

    FragmentInputUnpack unpacked_frag_;
    FragmentScale scale_frag_;

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
        for (int mma_n_iter = 0; mma_n_iter < kWarpIterationsAlongN; ++mma_n_iter) {
            super_scale_frag[mma_n_iter] = pointer_super_scale_[mma_n_iter * InstructionShape::kN]; // bank conflict
            code_scale_frag[mma_n_iter] = pointer_code_scale_[mma_n_iter * InstructionShape::kN];
            code_zp_frag[mma_n_iter] = pointer_code_zp_[mma_n_iter * InstructionShape::kN];
        }
    }

    /// Group-wise params, need to load multiple times
    CUTLASS_DEVICE
    void load(FragmentLocalScale& local_scale_frag) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kWarpIterationsAlongN; ++mma_n_iter) {
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
                    int tb_offset_k,
                    int warp_k_compute_offset) {
        int stage = tb_offset_k / 64;

        if (warp_k_compute_offset == 0) {
            unpacked_frag_ = Uint2Converter::convert(input_frag, code_scale_frag, code_zp_frag);
        }

        if (warp_k_compute_offset == 0) {
            // special for TileRows = 64
            int local_scale_shift = (((tb_offset_k / kGroupSize) + 1) & 1) * 4;

            if constexpr (platform::is_same<ElementOperand_, bfloat16_t>::value) {
                constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
                constexpr uint32_t MASK = 0x000f000f;
                constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

                constexpr uint32_t BF16_BIAS = 0xC300C300;
                constexpr uint32_t BF16_ONE = 0x3F803F80;

                __nv_bfloat162* scale_ptr = reinterpret_cast<__nv_bfloat162 *>(&scale_frag_);
                __nv_bfloat162 const* super_scale_ptr = reinterpret_cast<__nv_bfloat162 const*>(&super_scale_frag);

                uint32_t const* local_scale_ptr = reinterpret_cast<uint32_t const*>(&local_scale_frag);

                static_assert(FragmentLocalScale::kElements % 4 == 0, "");

                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < FragmentLocalScale::kElements / 4; ++i) {
                    int i4s = local_scale_ptr[i];

                    // unpack: 0, 2
                    i4s >>= local_scale_shift;
                    int32_t unpack0 = lop3<immLut>(i4s, MASK, I4s_TO_BF16s_MAGIC_NUM);
                    // unpack: 1, 3
                    i4s >>= 8;
                    int32_t unpack1 = lop3<immLut>(i4s, MASK, I4s_TO_BF16s_MAGIC_NUM);

                    nv_bfloat162 scale0 = __hfma2(*reinterpret_cast<nv_bfloat162*>(&unpack0),
                                                  *reinterpret_cast<const nv_bfloat162*>(&BF16_ONE),
                                                  *reinterpret_cast<const nv_bfloat162*>(&BF16_BIAS));
                    nv_bfloat162 scale1 = __hfma2(*reinterpret_cast<nv_bfloat162*>(&unpack1),
                                                  *reinterpret_cast<const nv_bfloat162*>(&BF16_ONE),
                                                  *reinterpret_cast<const nv_bfloat162*>(&BF16_BIAS));

                    // swap
                    nv_bfloat16 tmp = scale0.y;
                    scale0.y = scale1.x;
                    scale1.x = tmp;

                    scale_ptr[2 * i] = __hmul2(scale0, super_scale_ptr[2 * i]);
                    scale_ptr[2 * i + 1] = __hmul2(scale1, super_scale_ptr[2 * i + 1]);
                }
            } else {
                constexpr uint32_t kLocalScaleMask = 0xf;

                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < FragmentLocalScale::kElements; ++i) {
                    int32_t shifted_value = (static_cast<int32_t>(local_scale_frag[i]) >> local_scale_shift) & kLocalScaleMask;
                    scale_frag_[i] = static_cast<ElementCompute>(shifted_value) * super_scale_frag[i];
                }
            }
        }

        int offset = warp_k_compute_offset * ArchMmaOperator::FragmentB::kElements;
        const int kOutputColumns = FragmentOutput::kElements / kWarpIterationsAlongN;

        // reorder: [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
        int mapped_offset = (warp_k_compute_offset % 2) == 0 ? 0 : (-kOutputColumns + 1);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kWarpIterationsAlongN; ++mma_n_iter) {

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kOutputColumns; ++j) {
                // After applying LOP3 optimizations for performance, the B operand requires data rearrangement.
                int mapped_idx = mma_n_iter * kExpansionFactor * kOutputColumns + offset + 2 * j + mapped_offset;
                ElementCompute scaled_value =
                    static_cast<ElementCompute>(unpacked_frag_[mapped_idx]) * scale_frag_[mma_n_iter];
                output_frag[mma_n_iter * kOutputColumns + j] = static_cast<ElementOperand>(scaled_value);
            }
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
