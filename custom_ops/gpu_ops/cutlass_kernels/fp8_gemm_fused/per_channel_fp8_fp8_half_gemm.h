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

#pragma once

#include "fp8_common.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
// clang-format on

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CtaShape,
    typename WarpShape, typename InstructionShape, int Stages>
struct DeviceGemmFp8RowwiseSm89
{
    using ElementInput = typename std::conditional_t<
        std::is_same_v<ElementType, phi::dtype::float8_e4m3fn>,
        cutlass::float_e4m3_t,
        cutlass::float_e5m2_t>;
    using ElementA = ElementInput;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementInput;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;


    using ElementOutput =
        typename std::conditional_t<std::is_same_v<OutElementType, phi::dtype::bfloat16>,
                                    cutlass::bfloat16_t,
                                    cutlass::half_t>;

    using ElementC = ElementOutput;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using LayoutOutput = cutlass::layout::RowMajor;
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using ElementAccumulator = AccumElementType;
    using ElementComputeEpilogueScale = float;
    using ArchTag = cutlass::arch::Sm89;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // Number of epilogue stages in EVT
    static constexpr int EVTEpilogueStages = 1;

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<CtaShape, WarpShape, ElementOutput,
        AlignmentC, EVTEpilogueStages>;

    // Definition of EVT
    using accSrc = cutlass::epilogue::threadblock::VisitorAccFetch;

    using ComputeBScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementComputeEpilogueScale,
        ElementComputeEpilogueScale, cutlass::FloatRoundStyle::round_to_nearest>;
    using bScaleSrc = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementComputeEpilogueScale,
        cute::Stride<cute::_0, cute::_1, cute::_0>>;
    using EpilogueBScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeBScale, accSrc, bScaleSrc>;

    using ComputeAScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementComputeEpilogueScale,
        ElementComputeEpilogueScale, cutlass::FloatRoundStyle::round_to_nearest>;
    using aScaleSrc = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, ElementComputeEpilogueScale,
        cute::Stride<cute::_0, cute::_0, cute::_0>>;
    using EpilogueAScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScale, EpilogueBScale, aScaleSrc>;

    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, ElementC,
        cute::Stride<cute::_0, cute::_1, cute::_0>  // StrideMNL
    >;

    using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, ElementC, ElementComputeEpilogueScale,
        cutlass::FloatRoundStyle::round_to_nearest
    >;

    using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
        Compute0,
        EpilogueAScale,
        Bias>;

    using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementOutput,
        cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1,cute:: _0>>;
    using EpilogueStore = cutlass::epilogue::threadblock::Sm80EVT<dTar, EVTCompute0>;

    using EpilogueOp = EpilogueStore;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC, ElementAccumulator, ElementComputeEpilogueScale, OperatorClass, ArchTag, CtaShape,
        WarpShape, InstructionShape, EpilogueOp, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages,
        cutlass::arch::OpMultiplyAddFastAccum, EVTEpilogueStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
