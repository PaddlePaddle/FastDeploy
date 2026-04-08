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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

enum WintQuantMethod {
  kNone = 0,
  kWeightOnlyInt8 = 1,
  kWeightOnlyInt4 = 2,
  kWeightOnlyInt25 = 3,
  kWeightOnlyInt2 = 4
};

// Convert CUDA data type to cutlass data type
template <typename T> struct CutlassDataType {
  using Type = T;
};

template <> struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <> struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

template <typename ElementT, WintQuantMethod Method> struct WintQuantTraits;

template <typename ElementT>
struct WintQuantTraits<ElementT, WintQuantMethod::kNone> {
  using WeightType = ElementT;
  using MmaKernelType = typename CutlassDataType<ElementT>::Type;
  using MmaWeightType = typename CutlassDataType<ElementT>::Type;

  static constexpr WintQuantMethod kQuantMethod = WintQuantMethod::kNone;

  struct Arguments {};

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <typename ElementT>
struct WintQuantTraits<ElementT, WintQuantMethod::kWeightOnlyInt8> {
  using WeightType = uint8_t;
  using MmaKernelType = uint8_t;
  using MmaWeightType = uint8_t;

  static constexpr WintQuantMethod kQuantMethod =
      WintQuantMethod::kWeightOnlyInt8;

  struct Arguments {};

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <typename ElementT>
struct WintQuantTraits<ElementT, WintQuantMethod::kWeightOnlyInt4> {
  using WeightType = cutlass::uint4b_t;
  using MmaKernelType = cutlass::uint4b_t;
  using MmaWeightType = cutlass::uint4b_t;

  static constexpr WintQuantMethod kQuantMethod =
      WintQuantMethod::kWeightOnlyInt4;

  struct Arguments {};

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <typename ElementT>
struct WintQuantTraits<ElementT, WintQuantMethod::kWeightOnlyInt25> {
  using WeightType = uint16_t;
  using MmaKernelType = typename CutlassDataType<ElementT>::Type;
  using MmaWeightType = typename CutlassDataType<ElementT>::Type;

  static constexpr WintQuantMethod kQuantMethod =
      WintQuantMethod::kWeightOnlyInt25;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kNumPackedValues = 7;
  static constexpr int32_t kPackedSize = 10;

  struct Arguments {};

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) {
    return dim * kPackedSize / kGroupSize;
  }
};

template <typename ElementT>
struct WintQuantTraits<ElementT, WintQuantMethod::kWeightOnlyInt2> {
  using WeightType = uint8_t;
  using MmaKernelType = cutlass::uint2b_t;
  using MmaWeightType = typename CutlassDataType<ElementT>::Type;

  static constexpr WintQuantMethod kQuantMethod =
      WintQuantMethod::kWeightOnlyInt2;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kNumPackedValues = 4;
  static constexpr int32_t kPackedSize = 16;

  using LocalScaleType = uint4b_t;
  using CodeScaleZpType = float;

  struct Arguments {
    uint8_t *local_scale_ptr; // quanted 4-bits
    float *code_scale_ptr;
    float *code_zp_ptr;
  };

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) {
    return dim * kPackedSize / kGroupSize;
  }
};

} // namespace cutlass
