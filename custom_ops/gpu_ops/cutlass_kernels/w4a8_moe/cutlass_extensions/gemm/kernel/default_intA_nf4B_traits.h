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

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_kernels/w4a8_moe/cutlass_extensions/arch/mma.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename InAType, typename IntBType, typename OutType, typename arch, typename Enable = void>
struct Int8Nf4GemmArchTraits {
};

template<typename arch>
struct Int8Nf4GemmArchTraits<float, float, float, arch> {
    static constexpr int Stages = 2;
    using OperatorClass         = cutlass::arch::OpClassSimt;
    using AccType               = float;
    using LayoutB               = cutlass::layout::RowMajor;

    static constexpr int ElementsPerAccessA = 1;
    static constexpr int ElementsPerAccessB = 1;
    static constexpr int ElementsPerAccessC = 1;
    static constexpr int ThreadblockK       = 8;
    using InstructionShape                  = cutlass::gemm::GemmShape<1, 1, 1>;

    using Operator = cutlass::arch::OpMultiplyAdd;
};

// ======================= Ampere Traits ==============================
template<typename IntAType, typename IntBType, typename OutType>
struct Int8Nf4GemmArchTraits<IntAType,
                          IntBType,
                          OutType,
                          cutlass::arch::Sm80,
                          typename cutlass::platform::enable_if<
                            cutlass::platform::is_same<IntBType, int8_t>::value ||
                            cutlass::platform::is_same<IntBType, cutlass::uint4b_t>::value>::type> {
private:
    using LayoutDetails = LayoutDetailsB<IntBType, cutlass::arch::Sm80>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType       = int32_t;
    using LayoutB       = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<IntAType>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<OutType>::value;
    // static_assert(cutlass::platform::is_same<InType, int8_t>::value,
    //               "input type must be int8_t");
    // static_assert((ElementsPerAccessA == 16), "=====");
    // static_assert((ElementsPerAccessB == 16), "=====");
    // static_assert((ElementsPerAccessC == 8), "=====");
    using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 32>;
    // using InstructionShape                  = cutlass::gemm::GemmShape<16, 8, 16>;
    using Operator = typename LayoutDetails::Operator;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
