// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "cutlass/arch/mma_sm80.h"

namespace cutlass {
namespace arch {

template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};


}
}
