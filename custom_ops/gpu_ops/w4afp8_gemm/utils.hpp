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

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename T>
struct PackedHalf;

template <>
struct PackedHalf<cutlass::half_t> {
  using Type = __half2;
};

template <>
struct PackedHalf<cutlass::bfloat16_t> {
  using Type = nv_bfloat162;
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(
    Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <int numel>
__forceinline__ __device__ void convert_c4_2_fp8(const int32_t *src,
                                                 int32_t *dst1,
                                                 int32_t *dst2) {
#pragma unroll
  for (int i = 0; i < numel; ++i) {
    uint32_t head1 = src[i] & 0x80808080;
    dst1[i] = (src[i] >> 4) & 0x07070707;
    dst1[i] = dst1[i] | head1;
    uint32_t head2 = (src[i] & 0x08080808) << 4;
    dst2[i] = src[i] & 0x07070707;
    dst2[i] = dst2[i] | head2;
  }
}

template <int wg_wait = 0,
          bool arrive = true,
          bool commit = true,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename Tensor3,
          typename TiledMma,
          typename ThrCopyA,
          typename TiledCopyA>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma,
                                     Tensor0 &tCrA,
                                     Tensor1 &tCsA,
                                     Tensor2 const &tCrB,
                                     Tensor3 &tCrC,
                                     TiledCopyA const &tiled_copy_A,
                                     ThrCopyA const &thr_copy_A) {
  constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator,
                                           typename TiledMma::FrgTypeA>::value;
  Tensor tCrA1 = make_tensor<cutlass::float_e4m3_t>(tCrA.layout());
  Tensor tCrA2 = make_tensor<cutlass::float_e4m3_t>(tCrA.layout());
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (arrive) {
    warpgroup_arrive();
  }
  constexpr int numel = decltype(size(tCrA(_, _, 0)))::value / 4;
  Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
  cute::copy(tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));

  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    if (k_block < size<2>(tCrA) - 1) {
      cute::copy(tiled_copy_A,
                 tCsA(_, _, k_block + 1),
                 tCrA_copy_view(_, _, k_block + 1));
    }
    int32_t *tCrA_data =
        reinterpret_cast<int32_t *>(tCrA(_, _, k_block).data());
    int32_t *tCrA1_data =
        reinterpret_cast<int32_t *>(tCrA1(_, _, k_block).data());
    int32_t *tCrA2_data =
        reinterpret_cast<int32_t *>(tCrA2(_, _, k_block).data());
    convert_c4_2_fp8<numel>(tCrA_data, tCrA1_data, tCrA2_data);

    cute::gemm(tiled_mma, tCrA1(_, _, k_block), tCrB(_, _, 2 * k_block), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    cute::gemm(
        tiled_mma, tCrA2(_, _, k_block), tCrB(_, _, 2 * k_block + 1), tCrC);
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
  }
}
