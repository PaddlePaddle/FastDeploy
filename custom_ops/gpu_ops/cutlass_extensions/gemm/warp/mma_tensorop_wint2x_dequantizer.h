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

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<bfloat16_t> {
    using Type = __nv_bfloat16;
    using DualType = __nv_bfloat162;
};

template <>
struct DataTypeTraits<half_t> {
    using Type = __half;
    using DualType = __half2;
};

template <typename T, int N, typename Enable = void>
struct LocalScaleConverter {
    using FragmentSource = Array<uint8_t, N>;
    using FragmentResult = Array<T, N>;

    CUTLASS_DEVICE
    static void Apply(FragmentSource const& local_scale_frag,
                      FragmentResult const& super_scale_frag,
                      FragmentResult& scale_frag,
                      int shift_bit) {
        constexpr uint32_t kLocalScaleMask = 0xf;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            int32_t shifted_value = (static_cast<int32_t>(local_scale_frag[i]) >> shift_bit) & kLocalScaleMask;
            scale_frag[i] = static_cast<T>(shifted_value) * super_scale_frag[i];
        }
    }
};

template <int N>
struct LocalScaleConverter<half_t, N, typename platform::enable_if<N % 4 == 0>::type> {
    using FragmentSource = Array<uint8_t, N>;
    using FragmentResult = Array<half_t, N>;

    CUTLASS_DEVICE
    static void Apply(FragmentSource const& local_scale_frag,
                      FragmentResult const& super_scale_frag,
                      FragmentResult& scale_frag,
                      int shift_bit) {
        constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        constexpr uint32_t MASK = 0x000f000f;
        // 2^10 = 1024
        constexpr uint32_t I4s_TO_FP16s_MAGIC_NUM = 0x64006400;

        // -2^10 = -1024
        constexpr uint32_t FP16_BIAS = 0xE400E400;
        // 1.0
        constexpr uint32_t FP16_ONE = 0x3C003C00;

        __half2* scale_ptr = reinterpret_cast<__half2 *>(&scale_frag);
        __half2 const* super_scale_ptr = reinterpret_cast<__half2 const*>(&super_scale_frag);

        uint32_t const* local_scale_ptr = reinterpret_cast<uint32_t const*>(&local_scale_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 4; ++i) {
            int i4s = local_scale_ptr[i] >> shift_bit;

            // unpack: 0, 1
            int32_t low = __byte_perm(i4s, i4s, 0xF1F0);
            int32_t unpack0 = lop3<immLut>(low, MASK, I4s_TO_FP16s_MAGIC_NUM);
            // unpack: 2, 3
            int32_t high = __byte_perm(i4s, i4s, 0xF3F2);
            int32_t unpack1 = lop3<immLut>(high, MASK, I4s_TO_FP16s_MAGIC_NUM);

            __half2 scale0 = __hfma2(*reinterpret_cast<__half2*>(&unpack0),
                                     *reinterpret_cast<const __half2*>(&FP16_ONE),
                                     *reinterpret_cast<const __half2*>(&FP16_BIAS));
            __half2 scale1 = __hfma2(*reinterpret_cast<__half2*>(&unpack1),
                                     *reinterpret_cast<const __half2*>(&FP16_ONE),
                                     *reinterpret_cast<const __half2*>(&FP16_BIAS));

            scale_ptr[2 * i] = __hmul2(scale0, super_scale_ptr[2 * i]);
            scale_ptr[2 * i + 1] = __hmul2(scale1, super_scale_ptr[2 * i + 1]);
        }
    }
};

template <int N>
struct LocalScaleConverter<bfloat16_t, N, typename platform::enable_if<N % 4 == 0>::type> {
    using FragmentSource = Array<uint8_t, N>;
    using FragmentResult = Array<bfloat16_t, N>;

    CUTLASS_DEVICE
    static void Apply(FragmentSource const& local_scale_frag,
                      FragmentResult const& super_scale_frag,
                      FragmentResult& scale_frag,
                      int shift_bit) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        constexpr uint32_t immLut = (0xF0 & 0xCC) | 0xAA;
        constexpr uint32_t MASK = 0x000F000F;
        constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

        constexpr uint32_t BF16_BIAS = 0xC300C300;
        constexpr uint32_t BF16_ONE = 0x3F803F80;

        __nv_bfloat162* scale_ptr = reinterpret_cast<__nv_bfloat162 *>(&scale_frag);
        __nv_bfloat162 const* super_scale_ptr = reinterpret_cast<__nv_bfloat162 const*>(&super_scale_frag);

        uint32_t const* local_scale_ptr = reinterpret_cast<uint32_t const*>(&local_scale_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / 4; ++i) {
            int i4s = local_scale_ptr[i] >> shift_bit;

            // unpack: 0, 1
            int32_t low = __byte_perm(i4s, i4s, 0xF1F0);
            int32_t unpack0 = lop3<immLut>(low, MASK, I4s_TO_BF16s_MAGIC_NUM);
            // unpack: 2, 3
            int32_t high = __byte_perm(i4s, i4s, 0xF3F2);
            int32_t unpack1 = lop3<immLut>(high, MASK, I4s_TO_BF16s_MAGIC_NUM);

            nv_bfloat162 scale0 = __hfma2(*reinterpret_cast<nv_bfloat162*>(&unpack0),
                                          *reinterpret_cast<const nv_bfloat162*>(&BF16_ONE),
                                          *reinterpret_cast<const nv_bfloat162*>(&BF16_BIAS));
            nv_bfloat162 scale1 = __hfma2(*reinterpret_cast<nv_bfloat162*>(&unpack1),
                                          *reinterpret_cast<const nv_bfloat162*>(&BF16_ONE),
                                          *reinterpret_cast<const nv_bfloat162*>(&BF16_BIAS));

            scale_ptr[2 * i] = __hmul2(scale0, super_scale_ptr[2 * i]);
            scale_ptr[2 * i + 1] = __hmul2(scale1, super_scale_ptr[2 * i + 1]);
        }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
        // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
        // numerous conversion instructions in GEMM main loop.
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

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    static constexpr int kNumPacks = sizeof_bits<uint8_t>::value / sizeof_bits<ElementB>::value;
    static constexpr int kUnpackFactor = MmaOperator::FragmentB::kElements / (kWarpIterationsAlongN * kNumPacks);
    static constexpr int kUnpackInterval = kExpansionFactor / kUnpackFactor;

    /// Unpack 4 uint2b_t values compreseed in a uint8_t to floating points.
    using Uint2Converter = FastInterleavedAndBiasedNumericArrayConverter<
        ElementOperand, ElementB, MmaOperator::FragmentB::kElements / kUnpackFactor>;
    using FragmentInputUnpack = typename Uint2Converter::result_type;

    /// Fragment to hold internal scales before Mma
    using FragmentScale = Array<ElementOperand, FragmentLocalScale::kElements>;

    /// Fragment of dequantized B
    using FragmentOutput = Array<ElementOperand, MmaOperator::FragmentB::kElements / kExpansionFactor>;

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

    //FragmentInputUnpack unpacked_frag_;
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
        if constexpr (kUnpackInterval != 1) {
            // unsupport now
            arch::device_breakpoint();
        }

        typename Uint2Converter::source_type source_frag;

        int in_offset = warp_k_compute_offset * kUnpackInterval;

        uint8_t const* ptr_input = reinterpret_cast<uint8_t const*>(&input_frag);
        uint8_t* ptr_source = reinterpret_cast<uint8_t *>(&source_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kWarpIterationsAlongN; ++mma_n_iter) {
            ptr_source[mma_n_iter] = ptr_input[mma_n_iter * kUnpackFactor + in_offset];
        }
        FragmentInputUnpack unpacked_frag = Uint2Converter::convert(source_frag, code_scale_frag, code_zp_frag);

        // dequantize local_scale
        if (warp_k_compute_offset == 0) {
            using LocalScaleConverter = detail::LocalScaleConverter<ElementOperand, FragmentLocalScale::kElements>;

            // special for TileRows = 64
            int local_scale_shift = (((tb_offset_k / kGroupSize) + 1) & 1) * 4;
            LocalScaleConverter::Apply(local_scale_frag, super_scale_frag, scale_frag_, local_scale_shift);
        }

        // unscale
        // After applying LOP3 optimizations for performance, the B operand requires data rearrangement.
        // reorder: [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
        const int kWarpIterationsAlongK = FragmentOutput::kElements / kWarpIterationsAlongN;

        using Type = typename detail::DataTypeTraits<ElementOperand>::Type;
        using DualType = typename detail::DataTypeTraits<ElementOperand>::DualType;

        Type* output_ptr = reinterpret_cast<Type *>(&output_frag);
        DualType const* unpacked_ptr = reinterpret_cast<DualType const*>(&unpacked_frag);
        DualType const* scale_ptr = reinterpret_cast<DualType const*>(&scale_frag_);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < kWarpIterationsAlongN; mma_n_iter += 2) {
            int mapped_idx_base = (mma_n_iter / 2) * kWarpIterationsAlongK;

            DualType scalex2 = scale_ptr[mma_n_iter / 2];

            CUTLASS_PRAGMA_UNROLL
            for (int mma_k_iter = 0; mma_k_iter < kWarpIterationsAlongK; ++mma_k_iter) {
                DualType unpacked_valuex2 = unpacked_ptr[mapped_idx_base + mma_k_iter];
                DualType scaled_value = __hmul2(unpacked_valuex2, scalex2);
                output_ptr[mma_n_iter * kWarpIterationsAlongK + mma_k_iter] = scaled_value.x;
                output_ptr[(mma_n_iter + 1) * kWarpIterationsAlongK + mma_k_iter] = scaled_value.y;
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
