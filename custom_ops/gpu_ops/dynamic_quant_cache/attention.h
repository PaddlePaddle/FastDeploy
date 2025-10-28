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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor.hpp"
#include "paddle/extension.h"
#include "utils.hpp"

namespace dynamic_quant_cache {
struct Block_attn_params {
  void *__restrict__ q_input;
  uint8_t *__restrict__ cache_k_c2;
  uint8_t *__restrict__ cache_v_c2;

  void *__restrict__ cache_k_c16;
  void *__restrict__ cache_v_c16;

  void *__restrict__ attn_out;
  void *__restrict__ partition_attn_out;
  int *__restrict__ cu_seq_q;
  float *sums;
  float *maxs;
  int *seq_lens_encoder;
  int *seq_lens_decoder;
  int *block_table;
  int max_input_length;
  int head_num;
  int kv_head_num;
  int max_num_blocks_per_seq;
  int batch_size;
  int max_num_partitions;
  float inv_sqrt_dh;
  int data_num_per_block;
  int c16_remain_seq_len;
};

template <int kGqaGroupSize_,
          int kNWarps_,
          int kTileN_,
          int kHeadDim_,
          typename input_type_,
          typename output_type_,
          typename scale_type_>
struct Block_attn_kernel_traits {
  using ElementAccum = float;
  using input_type = input_type_;
  using output_type = output_type_;
  using scale_type = scale_type_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kGqaGroupSize = kGqaGroupSize_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kMinGemmM = 16;
  static constexpr int kBlockM =
      (kGqaGroupSize + kMinGemmM - 1) / kMinGemmM * kMinGemmM;
  static constexpr int kBlockSize = 64;
  static_assert(kGqaGroupSize <= 16);
  static constexpr int32_t kNWarps = kNWarps_;
  static constexpr int32_t kNReduceWarps = 4;
  static constexpr int32_t kNThreads = kNWarps * 32;
  static constexpr int32_t kNReduceThreads = kNReduceWarps * 32;
  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<8>, Int<64>>, Stride<Int<64>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutQK = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kBlockSize>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockSize>, Int<kHeadDim>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockSize>>{}, GenRowMajor{})));

  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using MMA_Atom_Arch =
      std::conditional_t<std::is_same_v<input_type, cutlass::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma = TiledMMA<MMA_Atom_Arch,
                            Layout<Shape<_1, Int<kNThreads / 32>, _1>>,
                            Tile<_16, Int<kNThreads / 32 * 16>, _16>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemLayoutAtom = Layout<Shape<Int<kNThreads / 8>, _8>, Stride<_8, _1>>;
  using GmemTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, input_type>{},
                               GmemLayoutAtom{},
                               Layout<Shape<_1, _8>>{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cutlass::uint128_t>, output_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, input_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, input_type>;
  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, input_type>;

  static constexpr int kShareMemSizeC2 =
      size(SmemLayoutQ{}) * sizeof(input_type);

  static constexpr int kShareMemSizeC16 =
      (size(SmemLayoutQ{}) + size(SmemLayoutKV{}) * 2) * sizeof(input_type);
};
}  // namespace dynamic_quant_cache
