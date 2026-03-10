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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "helper.h"
#include "mem_util.cuh"
#include "utils.cuh"

// FP8 scale divisor constant (for SM90+)
#if defined(__gfx942__)
constexpr float kFp8ScaleDivisorDS = 224.f;
#else
constexpr float kFp8ScaleDivisorDS = 448.f;
#endif

namespace ds_mla {

/**
 * FP8 scaled conversion utilities
 */
template <typename OutT, typename InT>
__device__ __forceinline__ OutT fp8_scaled_convert(InT src, float scale) {
  return static_cast<OutT>(static_cast<float>(src) / scale);
}

template <>
__device__ __forceinline__ uint8_t
fp8_scaled_convert<uint8_t, __nv_bfloat16>(__nv_bfloat16 src, float scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  float val = __bfloat162float(src) / scale;
  val = fminf(fmaxf(val, -448.0f), 448.0f);
  __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(val);
  return *reinterpret_cast<uint8_t*>(&fp8_val);
#else
  return 0;
#endif
}

template <>
__device__ __forceinline__ uint8_t
fp8_scaled_convert<uint8_t, half>(half src, float scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  float val = __half2float(src) / scale;
  val = fminf(fmaxf(val, -448.0f), 448.0f);
  __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(val);
  return *reinterpret_cast<uint8_t*>(&fp8_val);
#else
  return 0;
#endif
}

/**
 * DeepSeek MLA FP8 Cache Write Kernel
 *
 * Cache format (fp8_ds_mla - 656 bytes per token):
 * - First 512 bytes: quantized NoPE part (512 x fp8_e4m3)
 * - Next 16 bytes: scale factors (4 x float32, one per 128 fp8 values)
 * - Last 128 bytes: RoPE part (64 x bfloat16, not quantized)
 *
 * Thread organization:
 * - First 2 warps (64 threads): handle NoPE FP8 quantization
 * - Last 1 warp (32 threads): handle RoPE copy
 * - Total: 96 threads per block
 */
template <typename scalar_t>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,         // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,         // [num_tokens, pe_dim]
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size,
                                               // cache_entry_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    // stride per block in cache
    const int entry_stride,  // stride per token entry in cache
    const int kv_c_stride,   // stride for kv_c input
    const int k_pe_stride,   // stride for k_pe input
    const int kv_lora_rank,  // 512 for DS MLA
    const int pe_dim,        // 64 for DS MLA
    const int block_size     // number of tokens per cache block
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // Cast kv_cache to 16-bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;

    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);

    // RoPE values start after the packed 8-bit NoPE values and the 32-bit
    // scales Position: kv_lora_rank/2 (256 bytes in 16-bit units) + 8 (16 bytes
    // of scales in 16-bit units)
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;

    // Vectorized store of two 16-bit values
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);

  // Vectorized load of eight 16-bit values
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(static_cast<float>(vals[0])),
                                    fabsf(static_cast<float>(vals[1]))),
                              fmaxf(fabsf(static_cast<float>(vals[2])),
                                    fabsf(static_cast<float>(vals[3])))),
                        fmaxf(fmaxf(fabsf(static_cast<float>(vals[4])),
                                    fabsf(static_cast<float>(vals[5]))),
                              fmaxf(fabsf(static_cast<float>(vals[6])),
                                    fabsf(static_cast<float>(vals[7])))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, __shfl_xor_sync(0xFFFF, max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = fmaxf(max_abs / kFp8ScaleDivisorDS, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] = fp8_scaled_convert<uint8_t, scalar_t>(vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

/**
 * Standard MLA Cache Write Kernel (non-FP8)
 *
 * For auto/bf16/fp16 cache types
 */
template <typename scalar_t, typename cache_t>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank +
                                     // pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,
    const int entry_stride,
    const int kv_c_stride,
    const int k_pe_stride,
    const int kv_lora_rank,
    const int pe_dim,
    const int block_size,
    const float* scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  // Copy kv_c (NoPE part)
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const int64_t src_idx = token_idx * kv_c_stride + i;
    const int64_t dst_idx =
        block_idx * block_stride + block_offset * entry_stride + i;
    kv_cache[dst_idx] = static_cast<cache_t>(kv_c[src_idx]);
  }

  // Copy k_pe (RoPE part)
  for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
    const int64_t src_idx = token_idx * k_pe_stride + i;
    const int64_t dst_idx = block_idx * block_stride +
                            block_offset * entry_stride + kv_lora_rank + i;
    kv_cache[dst_idx] = static_cast<cache_t>(k_pe[src_idx]);
  }
}

/**
 * Indexer K Quantization and Cache Kernel
 *
 * Quantizes K values to FP8 and stores them in cache with scale factors
 * Cache layout: [quantized_k (head_dim bytes)] + [scales
 * (head_dim/quant_block_size * 4 bytes)]
 */
template <typename scalar_t>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    uint8_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,
    const int quant_block_size,  // typically 128
    const int cache_block_size,
    const int cache_stride,
    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  if (slot_idx < 0 || head_dim_idx >= head_dim) {
    return;
  }

  // Load 4 values at once using float2 (for bf16/fp16)
  float2 k_val = reinterpret_cast<const float2*>(
      k)[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);

  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(static_cast<float>(k_val_ptr[i])));
  }

  // Warp reduction to find max across quant_block_size elements
  for (int mask = 16; mask > 0; mask /= 2) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask));
  }

  float scale = fmaxf(amax, 1e-4f) / kFp8ScaleDivisorDS;

  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * head_dim + head_dim_idx;

  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8_scaled_convert<uint8_t, scalar_t>(k_val_ptr[i], scale);
  }

  // First thread in warp writes the scale
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride +
        cache_block_size * head_dim +
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

/**
 * Gather Indexer K from Quantized Cache Kernel
 *
 * Gathers and dequantizes K values from the cache
 */
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim/quant_block_size*4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,
    const int64_t token_stride,
    const int64_t head_dim,
    const int64_t block_stride,
    const int64_t cache_token_stride,
    const int64_t cache_block_size,
    const int num_blocks,
    const int num_tokens,
    const int quant_block_size) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;

  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  for (int iter = 0; iter < (batch_size + blockDim.x - 1) / blockDim.x;
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

  __syncwarp();

  if (head_idx >= head_dim || token_idx >= num_tokens) {
    return;
  }

  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
  const int block_id = block_table[batch_idx[threadIdx.y] * num_blocks +
                                   inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_id * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];

  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

/**
 * Prefill DS MLA Write Cache Kernel
 *
 * Writes prefill KV data to DS MLA cache format
 */
template <typename T, int VecSize = 1>
__global__ void prefill_ds_mla_cache_kernel(
    const T* __restrict__ kv_nope,   // [num_tokens, kv_num_heads * nope_size]
    const T* __restrict__ kv_pe,     // [num_tokens, kv_num_heads * pe_size]
    uint8_t* __restrict__ kv_cache,  // [num_blocks, kv_num_heads, block_size,
                                     // entry_size]
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    const int* __restrict__ seq_lens_decoder,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,  // 512 for DS MLA
    const int pe_size,    // 64 for DS MLA
    const int block_size,
    const int entry_size,  // 656 for DS MLA FP8
    const uint32_t elem_cnt) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t nope_hidden_size = kv_num_heads * nope_size;
  const uint32_t pe_hidden_size = kv_num_heads * pe_size;
  const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / hidden_size;
    const uint32_t bias = linear_index % hidden_size;
    const uint32_t ori_bi = batch_id_per_token[token_idx];

    if (seq_lens[ori_bi] == 0) continue;

    const uint32_t ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int* block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    if (bias < nope_hidden_size) {
      const uint32_t inner_bias = bias;
      const uint32_t hi = inner_bias / nope_size;
      const uint32_t h_bias = inner_bias % nope_size;

      // For DS MLA FP8, NoPE part goes to first 512 bytes
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * entry_size +
          hi * block_size * entry_size + block_offset * entry_size + h_bias;
      const uint32_t ori_idx = token_idx * nope_hidden_size + inner_bias;

      Load<T, VecSize>(&kv_nope[ori_idx], &src_vec);

      // Convert to FP8 and store
      for (int i = 0; i < VecSize; i++) {
        float val = static_cast<float>(src_vec.val[i]);
        val = fminf(fmaxf(val, -448.0f), 448.0f);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(val);
        kv_cache[tgt_idx + i] = *reinterpret_cast<uint8_t*>(&fp8_val);
#endif
      }
    } else {
      const uint32_t inner_bias = bias - nope_hidden_size;
      const uint32_t hi = inner_bias / pe_size;
      const uint32_t h_bias = inner_bias % pe_size;

      // RoPE part goes after NoPE (512 bytes) + scales (16 bytes)
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * entry_size +
          hi * block_size * entry_size + block_offset * entry_size + nope_size +
          16 + h_bias * 2;  // *2 for bf16
      const uint32_t ori_idx = token_idx * pe_hidden_size + inner_bias;

      Load<T, VecSize>(&kv_pe[ori_idx], &src_vec);

      // Copy RoPE without quantization (as bf16/fp16)
      T* tgt_ptr = reinterpret_cast<T*>(&kv_cache[tgt_idx]);
      for (int i = 0; i < VecSize; i++) {
        tgt_ptr[i] = src_vec.val[i];
      }
    }
  }
}

/**
 * Decode DS MLA Write Cache Kernel
 */
template <typename T, int VecSize = 1>
__global__ void decode_ds_mla_cache_kernel(
    const T* __restrict__ kv_nope,
    const T* __restrict__ kv_pe,
    uint8_t* __restrict__ kv_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    const int* __restrict__ seq_lens_encoder,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const int entry_size,
    const uint32_t elem_cnt) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t nope_hidden_size = kv_num_heads * nope_size;
  const uint32_t pe_hidden_size = kv_num_heads * pe_size;
  const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int start_token_idx = cu_seqlens_q[ori_bi];

    if (seq_lens_encoder[ori_bi] > 0) return;

    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int* block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;

    if (bias < nope_hidden_size) {
      const uint32_t inner_bias = bias;
      const uint32_t hi = inner_bias / nope_size;
      const uint32_t h_bias = inner_bias % nope_size;

      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * entry_size +
          hi * block_size * entry_size + block_offset * entry_size + h_bias;
      const uint32_t ori_idx = start_token_idx * nope_hidden_size + inner_bias;

      Load<T, VecSize>(&kv_nope[ori_idx], &src_vec);

      for (int i = 0; i < VecSize; i++) {
        float val = static_cast<float>(src_vec.val[i]);
        val = fminf(fmaxf(val, -448.0f), 448.0f);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8_e4m3 fp8_val = static_cast<__nv_fp8_e4m3>(val);
        kv_cache[tgt_idx + i] = *reinterpret_cast<uint8_t*>(&fp8_val);
#endif
      }
    } else {
      const uint32_t inner_bias = bias - nope_hidden_size;
      const uint32_t hi = inner_bias / pe_size;
      const uint32_t h_bias = inner_bias % pe_size;

      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * entry_size +
          hi * block_size * entry_size + block_offset * entry_size + nope_size +
          16 + h_bias * 2;
      const uint32_t ori_idx = start_token_idx * pe_hidden_size + inner_bias;

      Load<T, VecSize>(&kv_pe[ori_idx], &src_vec);

      T* tgt_ptr = reinterpret_cast<T*>(&kv_cache[tgt_idx]);
      for (int i = 0; i < VecSize; i++) {
        tgt_ptr[i] = src_vec.val[i];
      }
    }
  }
}

}  // namespace ds_mla
