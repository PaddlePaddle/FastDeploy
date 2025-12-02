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
#include "paddle/extension.h"
#include "utils.hpp"

namespace dynamic_quant_cache {
template <typename T,
          typename ScaleType,
          int kBlockSize,
          int kHeadDim,
          int kThreads,
          bool is_encoder>
__device__ void write_c2_cache_kernel(T *k_input,
                                      T *v_input,
                                      uint8_t *cache_k_c2,
                                      uint8_t *cache_v_c2,
                                      const int *cu_seq_k,
                                      const int *encoder_seqs_len,
                                      const int *decoder_seqs_len,
                                      const int *block_tables,
                                      const int64_t *prompt_lens,
                                      const int max_num_blocks_per_seq,
                                      const int data_num_per_block,
                                      const int c16_remain_seq_len,
                                      const int head_num,
                                      const int kv_head_num,
                                      const int bidb,
                                      const int bidh,
                                      const int block_idx) {
  using SmemLayoutAtomKV = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomKV{}, Shape<Int<kBlockSize>, Int<kHeadDim>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockSize>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using MMA_Atom_Arch =
      std::conditional_t<std::is_same_v<T, cutlass::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma = TiledMMA<MMA_Atom_Arch,
                            Layout<Shape<_1, Int<kThreads / 32>, _1>>,
                            Tile<_16, Int<kThreads / 32 * 16>, _16>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemLayoutAtom = Layout<Shape<Int<kThreads / 8>, _8>, Stride<_8, _1>>;
  using GmemTiledCopyKV =
      decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, T>{},
                               GmemLayoutAtom{},
                               Layout<Shape<_1, _8>>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  using pakc_half = typename PackedHalf<T>::Type;

  const int tidx = threadIdx.x;
  const int seq_len_encoder = encoder_seqs_len[bidb];
  const int seq_len_decoder = decoder_seqs_len[bidb];
  const int warp_idx = tidx / 32;
  const int lane_idx = tidx % 32;

  const int chunk_prefill_token_idx = seq_len_decoder / kBlockSize;
  const int token_idx = block_idx * kBlockSize + seq_len_decoder;

  if constexpr (is_encoder) {
    if (seq_len_encoder == 0 ||
        token_idx >= seq_len_encoder + seq_len_decoder) {
      return;
    }
  } else {
    if (seq_len_decoder == 0) {
      return;
    }
  }

  const int c16_cache_max_len = kBlockSize + c16_remain_seq_len;
  int c16_cache_len;
  int c2_cache_len;
  if constexpr (is_encoder) {
    const int prompt_len = prompt_lens[bidb];
    c16_cache_len = prompt_len < c16_cache_max_len
                        ? prompt_len
                        : c16_remain_seq_len + prompt_len % kBlockSize;
    c2_cache_len = prompt_len - c16_cache_len;
  } else {
    c16_cache_len =
        seq_len_decoder + 1 < c16_cache_max_len
            ? seq_len_decoder
            : c16_remain_seq_len + (seq_len_decoder + 1) % kBlockSize;
    c2_cache_len = seq_len_decoder - c16_cache_len;
  }

  const int remain_token = c2_cache_len - token_idx;

  if constexpr (is_encoder) {
    if (remain_token <= 0) {
      return;
    }
  }

  extern __shared__ char smem_[];

  const int load_idx =
      is_encoder
          ? ((cu_seq_k[bidb] + block_idx * kBlockSize) * kv_head_num + bidh) *
                kHeadDim
          : bidb * c16_cache_max_len * kv_head_num * kHeadDim + bidh * kHeadDim;

  const int stride_k = kHeadDim * kv_head_num;

  Tensor gK = make_tensor(make_gmem_ptr(k_input + load_idx),
                          Shape<Int<kBlockSize>, Int<kHeadDim>>{},
                          make_stride(stride_k, _1{}));

  Tensor gV = make_tensor(make_gmem_ptr(v_input + load_idx),
                          Shape<Int<kBlockSize>, Int<kHeadDim>>{},
                          make_stride(stride_k, _1{}));

  Tensor sK =
      make_tensor(make_smem_ptr(reinterpret_cast<T *>(smem_)), SmemLayoutKV{});

  Tensor sV = make_tensor(sK.data() + kHeadDim * kBlockSize, SmemLayoutKV{});

  uint32_t *cache_k_smem = reinterpret_cast<uint32_t *>(smem_);
  uint32_t *cache_v_smem =
      reinterpret_cast<uint32_t *>(smem_ + kHeadDim * kBlockSize * sizeof(T));

  Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});

  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

  Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);

  Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tKcK = gmem_thr_copy_KV.partition_S(cK);

  TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_thread_slice(tidx);

  copy(gmem_tiled_copy_KV, tKgK, tKsK, tKcK);

  cute::cp_async_fence();

  auto smem_tiles_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiles_copy_K.get_thread_slice(tidx);
  auto smem_tiles_copy_V =
      make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiles_copy_V.get_thread_slice(tidx);

  Tensor tSsK = smem_thr_copy_K.partition_S(sK);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
  Tensor tSrV_copy_view = smem_thr_copy_V.retile_D(tOrVt);

  cute::cp_async_wait<0>();
  __syncthreads();

  copy(gmem_tiled_copy_KV, tVgV, tVsV, tKcK);
  cute::cp_async_fence();

  constexpr float dequant_factor = 512.0f;
  const pakc_half fp8_dequant_factor =
      pakc_half(dequant_factor, dequant_factor);
  const pakc_half dequant_scale_factor =
      pakc_half(0.3333333333333f, 0.3333333333333f);
  constexpr float quant_factor = 1.0f / dequant_factor;
  constexpr float max_factor = 3.0f / dequant_factor;
  const pakc_half fp8_quant_factor = pakc_half(quant_factor, quant_factor);
  const pakc_half max_blound = pakc_half(max_factor, max_factor);

  // 求k的最大值
  constexpr int warp_per_scale = kHeadDim * kBlockSize / 32 / 2;
  __shared__ pakc_half s_max[warp_per_scale * kThreads / 32];
  __shared__ pakc_half s_min[warp_per_scale * kThreads / 32];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; j += 2) {
      int4 value1 =
          *reinterpret_cast<int4 *>(raw_pointer_cast(tKsK(_, j, i).data()));
      int4 value2 =
          *reinterpret_cast<int4 *>(raw_pointer_cast(tKsK(_, j + 1, i).data()));

      for (int k = 0; k < 4; ++k) {
        pakc_half cur_value = reinterpret_cast<pakc_half *>(&value1)[k];
        pakc_half next_value = reinterpret_cast<pakc_half *>(&value2)[k];
        pakc_half max_value = HalfMax<T>()(cur_value, next_value);
        pakc_half min_value = HalfMin<T>()(cur_value, next_value);

        pakc_half neigh_max_value = __shfl_xor_sync(uint32_t(-1), max_value, 8);
        pakc_half neigh_min_value = __shfl_xor_sync(uint32_t(-1), min_value, 8);
        max_value = HalfMax<T>()(max_value, neigh_max_value);
        min_value = HalfMin<T>()(min_value, neigh_min_value);

        neigh_max_value = __shfl_xor_sync(uint32_t(-1), max_value, 16);
        neigh_min_value = __shfl_xor_sync(uint32_t(-1), min_value, 16);
        max_value = HalfMax<T>()(max_value, neigh_max_value);
        min_value = HalfMin<T>()(min_value, neigh_min_value);
        if (lane_idx < 8) {
          const int scale_idx =
              warp_idx * warp_per_scale + lane_idx * 4 + k + i * 32 + j * 32;
          s_max[scale_idx] = max_value;
          s_min[scale_idx] = min_value;
        }
      }
    }
  }

  __syncthreads();

  pakc_half max_value = s_max[tidx];
  pakc_half min_value = s_min[tidx];

  for (int i = 0; i < kThreads / 32; ++i) {
    pakc_half cur_max = s_max[i * kThreads + tidx];
    pakc_half cur_min = s_min[i * kThreads + tidx];
    max_value = HalfMax<T>()(max_value, cur_max);
    min_value = HalfMin<T>()(min_value, cur_min);
  }

  __syncthreads();

  pakc_half dequant_scale = (max_value - min_value) * dequant_scale_factor;
  pakc_half dequant_scale_min = pakc_half(0.0001f, 0.0001f);
  dequant_scale = HalfMax<T>()(dequant_scale, dequant_scale_min);
  const pakc_half quant_scale_factor = pakc_half(1.0f, 1.0f);
  pakc_half quant_scale = __h2div(quant_scale_factor, dequant_scale);

  pakc_half quant_zp = -min_value * quant_scale;

  pakc_half *s_quant = s_max;
  pakc_half *s_zp = s_min;

  s_quant[tidx] = quant_scale;
  s_zp[tidx] = quant_zp;

  copy(smem_tiles_copy_K, tSsK, tSrK_copy_view);

  constexpr int scale_k_num = kHeadDim / 8;

  __syncthreads();
  // 量化k
  const int col = lane_idx % 4;

#pragma unroll
  for (int i = 0; i < scale_k_num / 2; i += 2) {
    uint32_t quant_c2_value = 0;
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        pakc_half cur_value =
            reinterpret_cast<pakc_half *>(tSrK(_, _, i + k).data())[j];
        pakc_half next_value =
            reinterpret_cast<pakc_half *>(tSrK(_, _, i + k).data())[j + 2];

        const int scale_idx = (i + k) * 8 + col + j * 4;
        pakc_half cur_quant_value = s_quant[scale_idx];
        pakc_half cur_quant_zp = s_zp[scale_idx];
        pakc_half next_quant_value = s_quant[scale_idx + 64];
        pakc_half next_quant_zp = s_zp[scale_idx + 64];

        cur_value = __hfma2_relu(cur_value, cur_quant_value, cur_quant_zp) *
                    fp8_quant_factor;
        next_value = __hfma2_relu(next_value, next_quant_value, next_quant_zp) *
                     fp8_quant_factor;

        cur_value = HalfMin<T>()(cur_value, max_blound);
        next_value = HalfMin<T>()(next_value, max_blound);

        uint32_t fp8_value =
            dynamic_quant_cache::Convert_to_fp8<T, cutlass::float_e4m3_t>()(
                reinterpret_cast<uint32_t *>(&cur_value)[0],
                reinterpret_cast<uint32_t *>(&next_value)[0]);

        quant_c2_value |= fp8_value;

        if (j == 1 && k == 1) {
          continue;
        }
        quant_c2_value = quant_c2_value << 2;
      }
    }
    cache_k_smem[i * (kThreads / 2) + tidx] = quant_c2_value;
  }

  __syncthreads();

  // 将k的反量化scale 写回到全局内存中
  pakc_half *dequant_scale_smem =
      reinterpret_cast<pakc_half *>(cache_k_smem + scale_k_num / 4 * kThreads);

  dequant_scale_smem[tidx] = dequant_scale * fp8_dequant_factor;

  pakc_half *dequant_zp_smem = dequant_scale_smem + kThreads;

  dequant_zp_smem[tidx] = quant_zp * dequant_scale;

  cute::cp_async_wait<0>();
  __syncthreads();

  copy(smem_tiles_copy_V, tOsVt, tSrV_copy_view);

  const int row_idx = tidx / 8;
  // 求v的最大值
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 2; ++i) {
      int4 value =
          *reinterpret_cast<int4 *>(raw_pointer_cast(tVsV(_, j, i).data()));
      pakc_half *value_ptr = reinterpret_cast<pakc_half *>(&value);
      pakc_half max_value = value_ptr[0];
      pakc_half min_value = value_ptr[0];
      for (int k = 1; k < 4; ++k) {
        max_value = HalfMax<T>()(max_value, value_ptr[k]);
        min_value = HalfMin<T>()(min_value, value_ptr[k]);
      }
      // 线程之间求最大值
      pakc_half neigh_max_value = __shfl_xor_sync(uint32_t(-1), max_value, 1);
      pakc_half neigh_min_value = __shfl_xor_sync(uint32_t(-1), min_value, 1);
      max_value = HalfMax<T>()(max_value, neigh_max_value);
      min_value = HalfMin<T>()(min_value, neigh_min_value);

      neigh_max_value = __shfl_xor_sync(uint32_t(-1), max_value, 2);
      neigh_min_value = __shfl_xor_sync(uint32_t(-1), min_value, 2);
      max_value = HalfMax<T>()(max_value, neigh_max_value);
      min_value = HalfMin<T>()(min_value, neigh_min_value);

      if (col == 0) {
        float max_value_f = max(float(max_value.x), float(max_value.y));
        float min_value_f = min(float(min_value.x), float(min_value.y));
        reinterpret_cast<T *>(
            s_max)[j * 16 + row_idx + i * 128 + tidx % 8 / 4 * 64] =
            T(max_value_f);
        reinterpret_cast<T *>(
            s_min)[j * 16 + row_idx + i * 128 + tidx % 8 / 4 * 64] =
            T(min_value_f);
      }
    }
  }

  __syncthreads();

  max_value = s_max[tidx];
  min_value = s_min[tidx];

  dequant_scale = (max_value - min_value) * dequant_scale_factor;
  dequant_scale = HalfMax<T>()(dequant_scale, dequant_scale_min);
  quant_scale = __h2div(quant_scale_factor, dequant_scale);

  quant_zp = -min_value * quant_scale;

  s_quant[tidx] = quant_scale;
  s_zp[tidx] = quant_zp;

  __syncthreads();

  for (int i = 0; i < 4; ++i) {
    uint32_t quant_c2_value = 0;

    for (int j = 0; j < 4; ++j) {
      pakc_half *value = reinterpret_cast<pakc_half *>(
          raw_pointer_cast(tOrVt(_, j, i).data()));
      const int scale_idx = i * 8 + j * 32 + col;
      value[0] = __hfma2_relu(value[0], s_quant[scale_idx], s_zp[scale_idx]) *
                 fp8_quant_factor;
      value[1] =
          __hfma2_relu(value[1], s_quant[scale_idx + 4], s_zp[scale_idx + 4]) *
          fp8_quant_factor;

      value[0] = HalfMin<T>()(value[0], max_blound);
      value[1] = HalfMin<T>()(value[1], max_blound);

      uint32_t fp8_value =
          dynamic_quant_cache::Convert_to_fp8<T, cutlass::float_e4m3_t>()(
              reinterpret_cast<uint32_t *>(value)[0],
              reinterpret_cast<uint32_t *>(value)[1]);

      quant_c2_value |= fp8_value;

      if (j == 3) {
        continue;
      }
      quant_c2_value = quant_c2_value << 2;
    }
    cache_v_smem[i * kThreads + tidx] = quant_c2_value;
  }

  __syncthreads();

  // 将反量化scale 写回到共享内存中
  dequant_scale_smem =
      reinterpret_cast<pakc_half *>(cache_v_smem + scale_k_num / 4 * kThreads);

  dequant_scale_smem[tidx] = dequant_scale * fp8_dequant_factor;

  dequant_zp_smem = dequant_scale_smem + kThreads;

  dequant_zp_smem[tidx] = quant_zp * dequant_scale;

  __syncthreads();

  // 将kv写回到全局内存中
  const int store_block_idx = is_encoder ? chunk_prefill_token_idx + block_idx
                                         : c2_cache_len / kBlockSize;
  const int *block_table = block_tables + bidb * max_num_blocks_per_seq;
  const int physical_block_number = block_table[store_block_idx];

  const int cache_offset =
      physical_block_number * kv_head_num * data_num_per_block +
      bidh * data_num_per_block;

  const int kPackSize = 16 / sizeof(uint8_t);

  for (int i = tidx * kPackSize; i < data_num_per_block;
       i += kThreads * kPackSize) {
    *reinterpret_cast<int4 *>(cache_k_c2 + cache_offset + i) =
        *reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(cache_k_smem) +
                                  i);
  }

  for (int i = tidx * kPackSize; i < data_num_per_block;
       i += kThreads * kPackSize) {
    *reinterpret_cast<int4 *>(cache_v_c2 + cache_offset + i) =
        *reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(cache_v_smem) +
                                  i);
  }

  if constexpr (!is_encoder) {
    // 最后将cache写到前面
    constexpr int kStorePackSize = 16 / sizeof(T);
    const int data_per_row = kHeadDim / kStorePackSize;
    const int row_idx = tidx / data_per_row;
    const int col_idx = tidx % data_per_row * kStorePackSize;
    const int all_rows = kThreads / data_per_row;

    const int src_load_idx = load_idx + col_idx;

    T *cache_k_c16 = const_cast<T *>(k_input);
    T *cache_v_c16 = const_cast<T *>(v_input);

#pragma unroll 4
    for (int i = row_idx; i < c16_remain_seq_len - 1; i += all_rows) {
      int4 src =
          *reinterpret_cast<int4 *>(cache_k_c16 + src_load_idx +
                                    (i + kBlockSize) * kv_head_num * kHeadDim);
      *reinterpret_cast<int4 *>(cache_k_c16 + src_load_idx +
                                i * kv_head_num * kHeadDim) = src;
    }

#pragma unroll 4
    for (int i = row_idx; i < c16_remain_seq_len - 1; i += all_rows) {
      int4 src =
          *reinterpret_cast<int4 *>(cache_v_c16 + src_load_idx +
                                    (i + kBlockSize) * kv_head_num * kHeadDim);
      *reinterpret_cast<int4 *>(cache_v_c16 + src_load_idx +
                                i * kv_head_num * kHeadDim) = src;
    }
  }
}
}  // namespace dynamic_quant_cache
