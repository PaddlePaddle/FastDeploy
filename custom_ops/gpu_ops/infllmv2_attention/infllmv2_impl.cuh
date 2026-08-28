// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace fastdeploy {

template <typename T>
__device__ __forceinline__ float InfLLMToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float InfLLMToFloat<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float InfLLMToFloat<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T InfLLMFromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half InfLLMFromFloat<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
InfLLMFromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ int InfLLMQueryPosition(int token_id,
                                                   int batch_id,
                                                   const int* seq_lens_decoder,
                                                   const int* cu_seqlens_q) {
  return seq_lens_decoder[batch_id] + token_id - cu_seqlens_q[batch_id];
}

template <typename T>
__device__ __forceinline__ float InfLLMReadPaged(const T* cache,
                                                 const int* block_tables,
                                                 int max_blocks_per_seq,
                                                 int batch_id,
                                                 int logical_position,
                                                 int kv_head,
                                                 int kv_heads,
                                                 int block_size,
                                                 int head_dim,
                                                 int dim) {
  const int logical_block = logical_position / block_size;
  const int block_offset = logical_position % block_size;
  const int physical_block =
      block_tables[batch_id * max_blocks_per_seq + logical_block];
  const int64_t offset =
      ((static_cast<int64_t>(physical_block) * kv_heads + kv_head) *
           block_size +
       block_offset) *
          head_dim +
      dim;
  return InfLLMToFloat(cache[offset]);
}

template <typename T>
__global__ void InfLLMV2UpdateCompressedKKernel(const T* key_cache,
                                                T* compressed_k,
                                                T* compressed_k2,
                                                const int* block_tables,
                                                const int* seq_lens_decoder,
                                                const int* batch_id_per_token,
                                                const int* cu_seqlens_q,
                                                int tokens,
                                                int batch_size,
                                                int physical_blocks,
                                                int max_blocks_per_seq,
                                                int kv_heads,
                                                int block_size,
                                                int head_dim,
                                                int kernel_size,
                                                int kernel_stride) {
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int scale_id = blockIdx.z;
  const int dim = threadIdx.x;
  if (token_id >= tokens || kv_head >= kv_heads || dim >= head_dim) {
    return;
  }

  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    return;
  }
  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int window_size = scale_id == 0 ? kernel_size : 4 * kernel_size;
  const int stride = scale_id == 0 ? kernel_stride : 4 * kernel_stride;
  if (position + 1 < window_size ||
      (position + 1 - window_size) % stride != 0) {
    return;
  }

  float sum = 0.0f;
  const int window_begin = position + 1 - window_size;
  for (int logical_position = window_begin; logical_position <= position;
       ++logical_position) {
    sum += InfLLMReadPaged(key_cache,
                           block_tables,
                           max_blocks_per_seq,
                           batch_id,
                           logical_position,
                           kv_head,
                           kv_heads,
                           block_size,
                           head_dim,
                           dim);
  }

  const int logical_block = position / block_size;
  const int physical_block =
      block_tables[batch_id * max_blocks_per_seq + logical_block];
  if (physical_block < 0 || physical_block >= physical_blocks) {
    return;
  }
  const int slots = block_size / stride;
  const int slot = (position / stride) % slots;
  const int64_t output_offset =
      ((static_cast<int64_t>(physical_block) * kv_heads + kv_head) * slots +
       slot) *
          head_dim +
      dim;
  T* output = scale_id == 0 ? compressed_k : compressed_k2;
  output[output_offset] = InfLLMFromFloat<T>(sum / window_size);
}

template <typename T>
__device__ __forceinline__ float InfLLMReadSummary(const T* compressed,
                                                   const int* block_tables,
                                                   int max_blocks_per_seq,
                                                   int batch_id,
                                                   int window_end,
                                                   int stride,
                                                   int slots,
                                                   int kv_head,
                                                   int kv_heads,
                                                   int head_dim,
                                                   int dim) {
  const int logical_block = window_end / (slots * stride);
  const int physical_block =
      block_tables[batch_id * max_blocks_per_seq + logical_block];
  const int slot = (window_end / stride) % slots;
  const int64_t offset =
      ((static_cast<int64_t>(physical_block) * kv_heads + kv_head) * slots +
       slot) *
          head_dim +
      dim;
  return InfLLMToFloat(compressed[offset]);
}

template <typename T>
__global__ void InfLLMV2CoarseLSETensorCoreSplitKernel(
    const T* query,
    const T* compressed_k2,
    const int* block_tables,
    const int* seq_lens_decoder,
    const int* batch_id_per_token,
    const int* cu_seqlens_q,
    float* coarse_partial_max,
    float* coarse_partial_sum,
    int tokens,
    int batch_size,
    int max_blocks_per_seq,
    int query_heads,
    int kv_heads,
    int block_size,
    int head_dim,
    int kernel_size,
    int kernel_stride,
    int coarse_splits) {
  constexpr int kTensorTile = 16;
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_id = blockIdx.z;
  const int lane = threadIdx.x;
  if (token_id >= tokens || kv_head >= kv_heads || split_id >= coarse_splits) {
    return;
  }
  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    if (lane < kTensorTile) {
      const int query_head = kv_head * kTensorTile + lane;
      const int partial_offset =
          (token_id * query_heads + query_head) * coarse_splits + split_id;
      coarse_partial_max[partial_offset] = -CUDART_INF_F;
      coarse_partial_sum[partial_offset] = 0.0f;
    }
    return;
  }

  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int visible_length = position + 1;
  const int coarse_kernel = 4 * kernel_size;
  const int coarse_stride = 4 * kernel_stride;
  const int coarse_slots = block_size / coarse_stride;
  const int windows =
      visible_length < coarse_kernel
          ? 0
          : (visible_length - coarse_kernel) / coarse_stride + 1;
  const int first_window = split_id * kTensorTile;
  const int window_count = max(0, min(kTensorTile, windows - first_window));
  extern __shared__ char coarse_tensor_shared[];
  T* shared_k = reinterpret_cast<T*>(coarse_tensor_shared);
  const size_t scores_offset =
      (static_cast<size_t>(kTensorTile) * head_dim * sizeof(T) +
       alignof(float) - 1) &
      ~(alignof(float) - 1);
  float* shared_scores =
      reinterpret_cast<float*>(coarse_tensor_shared + scores_offset);

  for (int index = lane; index < kTensorTile * head_dim; index += 32) {
    const int local_window = index / head_dim;
    const int dim = index % head_dim;
    if (local_window < window_count) {
      const int window = first_window + local_window;
      const int window_end = coarse_kernel - 1 + window * coarse_stride;
      shared_k[index] = InfLLMReadSummary(compressed_k2,
                                          block_tables,
                                          max_blocks_per_seq,
                                          batch_id,
                                          window_end,
                                          coarse_stride,
                                          coarse_slots,
                                          kv_head,
                                          kv_heads,
                                          head_dim,
                                          dim);
    } else {
      shared_k[index] = InfLLMFromFloat<T>(0.0f);
    }
  }
  __syncwarp();

  using namespace nvcuda;
  wmma::
      fragment<wmma::accumulator, kTensorTile, kTensorTile, kTensorTile, float>
          accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  const T* query_base = query + (static_cast<int64_t>(token_id) * query_heads +
                                 kv_head * kTensorTile) *
                                    head_dim;
  for (int dim = 0; dim < head_dim; dim += kTensorTile) {
    wmma::fragment<wmma::matrix_a,
                   kTensorTile,
                   kTensorTile,
                   kTensorTile,
                   T,
                   wmma::row_major>
        query_fragment;
    wmma::fragment<wmma::matrix_b,
                   kTensorTile,
                   kTensorTile,
                   kTensorTile,
                   T,
                   wmma::col_major>
        key_fragment;
    wmma::load_matrix_sync(query_fragment, query_base + dim, head_dim);
    wmma::load_matrix_sync(key_fragment, shared_k + dim, head_dim);
    wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
  }
  wmma::store_matrix_sync(
      shared_scores, accumulator, kTensorTile, wmma::mem_row_major);
  __syncwarp();

  if (lane < kTensorTile) {
    const int query_head = kv_head * kTensorTile + lane;
    const int score_base = lane * kTensorTile;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float row_max = -CUDART_INF_F;
    for (int local_window = 0; local_window < window_count; ++local_window) {
      row_max =
          fmaxf(row_max, shared_scores[score_base + local_window] * scale);
    }
    float row_sum = 0.0f;
    for (int local_window = 0; local_window < window_count; ++local_window) {
      row_sum +=
          expf(shared_scores[score_base + local_window] * scale - row_max);
    }
    const int partial_offset =
        (token_id * query_heads + query_head) * coarse_splits + split_id;
    coarse_partial_max[partial_offset] = row_max;
    coarse_partial_sum[partial_offset] = row_sum;
  }
}

__global__ void InfLLMV2CoarseLSECombineKernel(const float* coarse_partial_max,
                                               const float* coarse_partial_sum,
                                               const int* batch_id_per_token,
                                               float* coarse_lse,
                                               int tokens,
                                               int batch_size,
                                               int query_heads,
                                               int coarse_splits) {
  const int token_id = blockIdx.x;
  const int query_head = blockIdx.y;
  const int thread = threadIdx.x;
  if (token_id >= tokens || query_head >= query_heads) {
    return;
  }
  const int partial_base =
      (token_id * query_heads + query_head) * coarse_splits;
  float local_max = -CUDART_INF_F;
  for (int split = thread; split < coarse_splits; split += blockDim.x) {
    local_max = fmaxf(local_max, coarse_partial_max[partial_base + split]);
  }
  extern __shared__ float reduction[];
  float* maxima = reduction;
  float* sums = reduction + blockDim.x;
  maxima[thread] = local_max;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (thread < width) {
      maxima[thread] = fmaxf(maxima[thread], maxima[thread + width]);
    }
    __syncthreads();
  }
  const float row_max = maxima[0];
  float local_sum = 0.0f;
  if (row_max != -CUDART_INF_F) {
    for (int split = thread; split < coarse_splits; split += blockDim.x) {
      const float split_sum = coarse_partial_sum[partial_base + split];
      if (split_sum > 0.0f) {
        local_sum += split_sum *
                     expf(coarse_partial_max[partial_base + split] - row_max);
      }
    }
  }
  sums[thread] = local_sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (thread < width) {
      sums[thread] += sums[thread + width];
    }
    __syncthreads();
  }
  if (thread == 0) {
    const int batch_id = batch_id_per_token[token_id];
    coarse_lse[token_id * query_heads + query_head] =
        batch_id < 0 || batch_id >= batch_size
            ? -CUDART_INF_F
            : (row_max == -CUDART_INF_F ? 0.0f : row_max + logf(sums[0]));
  }
}

template <typename T>
__global__ void InfLLMV2CoarseLSEKernel(const T* query,
                                        const T* compressed_k2,
                                        const int* block_tables,
                                        const int* seq_lens_decoder,
                                        const int* batch_id_per_token,
                                        const int* cu_seqlens_q,
                                        float* coarse_lse,
                                        float* coarse_partial_max,
                                        float* coarse_partial_sum,
                                        int tokens,
                                        int batch_size,
                                        int max_blocks_per_seq,
                                        int query_heads,
                                        int kv_heads,
                                        int block_size,
                                        int head_dim,
                                        int kernel_size,
                                        int kernel_stride,
                                        int coarse_splits) {
  const int token_id = blockIdx.x;
  const int query_head = blockIdx.y;
  if (token_id >= tokens || query_head >= query_heads) {
    return;
  }
  const int batch_id = batch_id_per_token[token_id];
  const int partial_base =
      (token_id * query_heads + query_head) * coarse_splits;
  for (int split = threadIdx.x; split < coarse_splits; split += blockDim.x) {
    coarse_partial_max[partial_base + split] = -CUDART_INF_F;
    coarse_partial_sum[partial_base + split] = 0.0f;
  }
  if (batch_id < 0 || batch_id >= batch_size) {
    if (threadIdx.x == 0) {
      coarse_lse[token_id * query_heads + query_head] = -CUDART_INF_F;
    }
    return;
  }

  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int visible_length = position + 1;
  const int coarse_kernel = 4 * kernel_size;
  const int coarse_stride = 4 * kernel_stride;
  const int coarse_slots = block_size / coarse_stride;
  const int windows =
      visible_length < coarse_kernel
          ? 0
          : (visible_length - coarse_kernel) / coarse_stride + 1;
  const int kv_head = query_head / (query_heads / kv_heads);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float local_max = -CUDART_INF_F;
  float local_sum = 0.0f;
  for (int window = threadIdx.x; window < windows; window += blockDim.x) {
    const int window_end = coarse_kernel - 1 + window * coarse_stride;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      const int64_t query_offset =
          (static_cast<int64_t>(token_id) * query_heads + query_head) *
              head_dim +
          dim;
      dot += InfLLMToFloat(query[query_offset]) *
             InfLLMReadSummary(compressed_k2,
                               block_tables,
                               max_blocks_per_seq,
                               batch_id,
                               window_end,
                               coarse_stride,
                               coarse_slots,
                               kv_head,
                               kv_heads,
                               head_dim,
                               dim);
    }
    const float score = dot * scale;
    if (score > local_max) {
      local_sum = local_max == -CUDART_INF_F
                      ? 1.0f
                      : local_sum * expf(local_max - score) + 1.0f;
      local_max = score;
    } else {
      local_sum += expf(score - local_max);
    }
  }

  extern __shared__ float reduction[];
  float* maxima = reduction;
  float* sums = reduction + blockDim.x;
  maxima[threadIdx.x] = local_max;
  sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      const float other_max = maxima[threadIdx.x + width];
      const float merged_max = fmaxf(maxima[threadIdx.x], other_max);
      float merged_sum = 0.0f;
      if (merged_max != -CUDART_INF_F) {
        if (maxima[threadIdx.x] != -CUDART_INF_F) {
          merged_sum +=
              sums[threadIdx.x] * expf(maxima[threadIdx.x] - merged_max);
        }
        if (other_max != -CUDART_INF_F) {
          merged_sum +=
              sums[threadIdx.x + width] * expf(other_max - merged_max);
        }
      }
      maxima[threadIdx.x] = merged_max;
      sums[threadIdx.x] = merged_sum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    coarse_lse[token_id * query_heads + query_head] =
        windows == 0 ? 0.0f : maxima[0] + logf(sums[0]);
  }
}

template <typename T>
__global__ void InfLLMV2BlockScoreTensorCoreKernel(
    const T* query,
    const T* compressed_k,
    const int* block_tables,
    const int* seq_lens_decoder,
    const int* batch_id_per_token,
    const int* cu_seqlens_q,
    const float* coarse_lse,
    float* block_scores,
    int tokens,
    int batch_size,
    int max_blocks_per_seq,
    int query_heads,
    int kv_heads,
    int block_size,
    int head_dim,
    int kernel_size,
    int kernel_stride,
    int init_blocks,
    int local_blocks) {
  constexpr int kWarpsPerCTA = 4;
  constexpr int kTensorTile = 16;
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int logical_block = blockIdx.z * kWarpsPerCTA + warp_id;
  if (token_id >= tokens || kv_head >= kv_heads ||
      logical_block >= max_blocks_per_seq) {
    return;
  }
  const int output_offset =
      (token_id * kv_heads + kv_head) * max_blocks_per_seq + logical_block;
  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    if (lane == 0) {
      block_scores[output_offset] = -CUDART_INF_F;
    }
    return;
  }
  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int visible_length = position + 1;
  const int valid_blocks = (visible_length + block_size - 1) / block_size;
  if (logical_block >= valid_blocks) {
    if (lane == 0) {
      block_scores[output_offset] = -CUDART_INF_F;
    }
    return;
  }
  const int current_block = position / block_size;
  if (logical_block < init_blocks ||
      (logical_block <= current_block &&
       logical_block + local_blocks > current_block)) {
    if (lane == 0) {
      block_scores[output_offset] = CUDART_INF_F;
    }
    return;
  }

  const int fine_slots = block_size / kernel_stride;
  const int first_window = max(0, logical_block * fine_slots - 1);
  const int fine_windows =
      visible_length < kernel_size
          ? 0
          : (visible_length - kernel_size) / kernel_stride + 1;
  const int last_window = min(fine_windows, (logical_block + 1) * fine_slots);
  const int window_count = last_window - first_window;
  extern __shared__ char tensor_score_shared[];
  const size_t shared_k_bytes =
      static_cast<size_t>(kTensorTile) * head_dim * sizeof(T);
  const size_t scores_offset =
      (shared_k_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
  const size_t warp_bytes =
      scores_offset + kTensorTile * kTensorTile * sizeof(float);
  char* warp_shared = tensor_score_shared + warp_id * warp_bytes;
  T* shared_k = reinterpret_cast<T*>(warp_shared);
  float* shared_scores = reinterpret_cast<float*>(warp_shared + scores_offset);

  for (int index = lane; index < kTensorTile * head_dim; index += 32) {
    const int local_window = index / head_dim;
    const int dim = index % head_dim;
    if (local_window < window_count) {
      const int window = first_window + local_window;
      const int window_end = kernel_size - 1 + window * kernel_stride;
      shared_k[index] = InfLLMReadSummary(compressed_k,
                                          block_tables,
                                          max_blocks_per_seq,
                                          batch_id,
                                          window_end,
                                          kernel_stride,
                                          fine_slots,
                                          kv_head,
                                          kv_heads,
                                          head_dim,
                                          dim);
    } else {
      shared_k[index] = InfLLMFromFloat<T>(0.0f);
    }
  }
  __syncwarp();

  using namespace nvcuda;
  wmma::
      fragment<wmma::accumulator, kTensorTile, kTensorTile, kTensorTile, float>
          accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  const int query_head = kv_head * kTensorTile;
  const T* query_base =
      query +
      (static_cast<int64_t>(token_id) * query_heads + query_head) * head_dim;
  for (int dim = 0; dim < head_dim; dim += kTensorTile) {
    wmma::fragment<wmma::matrix_a,
                   kTensorTile,
                   kTensorTile,
                   kTensorTile,
                   T,
                   wmma::row_major>
        query_fragment;
    wmma::fragment<wmma::matrix_b,
                   kTensorTile,
                   kTensorTile,
                   kTensorTile,
                   T,
                   wmma::col_major>
        key_fragment;
    wmma::load_matrix_sync(query_fragment, query_base + dim, head_dim);
    wmma::load_matrix_sync(key_fragment, shared_k + dim, head_dim);
    wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
  }
  wmma::store_matrix_sync(
      shared_scores, accumulator, kTensorTile, wmma::mem_row_major);
  __syncwarp();

  float gqa_score = -CUDART_INF_F;
  if (lane < window_count) {
    gqa_score = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (int head = 0; head < kTensorTile; ++head) {
      const float lse = coarse_lse[token_id * query_heads + query_head + head];
      gqa_score += expf(shared_scores[head * kTensorTile + lane] * scale - lse);
    }
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    gqa_score =
        fmaxf(gqa_score, __shfl_down_sync(0xffffffff, gqa_score, delta));
  }
  if (lane == 0) {
    block_scores[output_offset] = gqa_score;
  }
}

template <typename T>
__global__ void InfLLMV2BlockScoreKernel(const T* query,
                                         const T* compressed_k,
                                         const int* block_tables,
                                         const int* seq_lens_decoder,
                                         const int* batch_id_per_token,
                                         const int* cu_seqlens_q,
                                         const float* coarse_lse,
                                         float* block_scores,
                                         int tokens,
                                         int batch_size,
                                         int max_blocks_per_seq,
                                         int query_heads,
                                         int kv_heads,
                                         int block_size,
                                         int head_dim,
                                         int kernel_size,
                                         int kernel_stride,
                                         int init_blocks,
                                         int local_blocks) {
  extern __shared__ char block_score_shared[];
  constexpr int kBlocksPerCTA = 4;
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int first_logical_block = blockIdx.z * kBlocksPerCTA;
  if (token_id >= tokens || kv_head >= kv_heads ||
      first_logical_block >= max_blocks_per_seq) {
    return;
  }
  const int output_base = (token_id * kv_heads + kv_head) * max_blocks_per_seq;
  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    if (threadIdx.x == 0) {
      for (int offset = 0; offset < kBlocksPerCTA; ++offset) {
        const int logical_block = first_logical_block + offset;
        if (logical_block < max_blocks_per_seq) {
          block_scores[output_base + logical_block] = -CUDART_INF_F;
        }
      }
    }
    return;
  }
  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int visible_length = position + 1;
  const int valid_blocks = (visible_length + block_size - 1) / block_size;
  const int current_block = position / block_size;
  const int fine_slots = block_size / kernel_stride;
  const int fine_windows =
      visible_length < kernel_size
          ? 0
          : (visible_length - kernel_size) / kernel_stride + 1;
  const int group_size = query_heads / kv_heads;
  const int group_head = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int query_head = kv_head * group_size + group_head;
  const int values_per_lane = (head_dim + 31) / 32;
  float query_values[8];
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    query_values[element] = 0.0f;
  }
  const int64_t query_offset =
      (static_cast<int64_t>(token_id) * query_heads + query_head) * head_dim;
  for (int element = 0; element < values_per_lane; ++element) {
    const int dim = lane + element * 32;
    if (dim < head_dim) {
      query_values[element] = InfLLMToFloat(query[query_offset + dim]);
    }
  }
  T* shared_k = reinterpret_cast<T*>(block_score_shared);
  const size_t shared_k_bytes =
      static_cast<size_t>(fine_slots + 1) * head_dim * sizeof(T);
  const size_t scores_offset =
      (shared_k_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
  float* gqa_scores =
      reinterpret_cast<float*>(block_score_shared + scores_offset);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float lse = coarse_lse[token_id * query_heads + query_head];

  for (int block_offset = 0; block_offset < kBlocksPerCTA; ++block_offset) {
    const int logical_block = first_logical_block + block_offset;
    if (logical_block >= max_blocks_per_seq) {
      break;
    }
    const int output_offset = output_base + logical_block;
    if (logical_block >= valid_blocks) {
      if (threadIdx.x == 0) {
        block_scores[output_offset] = -CUDART_INF_F;
      }
      continue;
    }
    if (logical_block < init_blocks ||
        (logical_block <= current_block &&
         logical_block + local_blocks > current_block)) {
      if (threadIdx.x == 0) {
        block_scores[output_offset] = CUDART_INF_F;
      }
      continue;
    }

    const int first_window = max(0, logical_block * fine_slots - 1);
    const int last_window = min(fine_windows, (logical_block + 1) * fine_slots);
    const int window_count = last_window - first_window;
    for (int index = threadIdx.x; index < window_count * head_dim;
         index += blockDim.x) {
      const int local_window = index / head_dim;
      const int dim = index % head_dim;
      const int window = first_window + local_window;
      const int window_end = kernel_size - 1 + window * kernel_stride;
      shared_k[index] = InfLLMReadSummary(compressed_k,
                                          block_tables,
                                          max_blocks_per_seq,
                                          batch_id,
                                          window_end,
                                          kernel_stride,
                                          fine_slots,
                                          kv_head,
                                          kv_heads,
                                          head_dim,
                                          dim);
    }
    __syncthreads();

    for (int local_window = 0; local_window < window_count; ++local_window) {
      float local_dot = 0.0f;
      for (int element = 0; element < values_per_lane; ++element) {
        const int dim = lane + element * 32;
        if (dim < head_dim) {
          local_dot += query_values[element] *
                       InfLLMToFloat(shared_k[local_window * head_dim + dim]);
        }
      }
      for (int delta = 16; delta > 0; delta >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, delta);
      }
      if (lane == 0) {
        gqa_scores[local_window * group_size + group_head] =
            expf(local_dot * scale - lse);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float best = -CUDART_INF_F;
      for (int local_window = 0; local_window < window_count; ++local_window) {
        float gqa_score = 0.0f;
        for (int head = 0; head < group_size; ++head) {
          gqa_score += gqa_scores[local_window * group_size + head];
        }
        best = fmaxf(best, gqa_score);
      }
      block_scores[output_offset] = best;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ uint32_t InfLLMOrderedFloatBits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return bits ^ ((static_cast<int32_t>(bits) < 0) ? 0xffffffffu : 0x80000000u);
}

__global__ void InfLLMV2TopKKernel(const float* block_scores,
                                   const int* seq_lens_decoder,
                                   const int* batch_id_per_token,
                                   const int* cu_seqlens_q,
                                   int* topk_indices,
                                   int* selected_counts,
                                   int tokens,
                                   int batch_size,
                                   int kv_heads,
                                   int max_blocks_per_seq,
                                   int capacity,
                                   int block_size,
                                   int topk,
                                   int dense_len,
                                   int local_blocks) {
  constexpr int kBlockThreads = 256;
  constexpr int kItemsPerThread = 8;
  using BlockRadixSort =
      cub::BlockRadixSort<uint64_t, kBlockThreads, kItemsPerThread>;
  __shared__ typename BlockRadixSort::TempStorage sort_storage;

  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  if (token_id >= tokens || kv_head >= kv_heads) {
    return;
  }
  int* selected = topk_indices + (token_id * kv_heads + kv_head) * capacity;
  for (int slot = threadIdx.x; slot < capacity; slot += blockDim.x) {
    selected[slot] = -1;
  }
  const int count_offset = token_id * kv_heads + kv_head;
  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    if (threadIdx.x == 0) {
      selected_counts[count_offset] = 0;
    }
    return;
  }
  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int visible_length = position + 1;
  const int valid_blocks = (visible_length + block_size - 1) / block_size;
  const int target =
      min(valid_blocks,
          visible_length < dense_len ? valid_blocks : topk + local_blocks);
  const float* scores =
      block_scores + (token_id * kv_heads + kv_head) * max_blocks_per_seq;

  uint64_t sort_keys[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int candidate = threadIdx.x * kItemsPerThread + item;
    if (candidate < valid_blocks) {
      const uint64_t score_key = InfLLMOrderedFloatBits(scores[candidate]);
      const uint64_t tie_key = 0xffffffffu - candidate;
      sort_keys[item] = (score_key << 32) | tie_key;
    } else {
      sort_keys[item] = 0;
    }
  }

  BlockRadixSort(sort_storage).SortDescending(sort_keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    const uint32_t candidate =
        0xffffffffu - static_cast<uint32_t>(sort_keys[item]);
    sort_keys[item] = rank < target ? candidate : UINT64_MAX;
  }

  BlockRadixSort(sort_storage).Sort(sort_keys);

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = threadIdx.x * kItemsPerThread + item;
    if (rank < target) {
      selected[rank] = static_cast<int>(sort_keys[item]);
    }
  }
  if (threadIdx.x == 0) {
    selected_counts[count_offset] = target;
  }
}

template <typename T>
__global__ void InfLLMV2SparseAttentionTensorCoreSplitKVKernel(
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* block_tables,
    const int* seq_lens_decoder,
    const int* batch_id_per_token,
    const int* cu_seqlens_q,
    const int* topk_indices,
    float* partial_acc,
    float* partial_max,
    float* partial_sum,
    int tokens,
    int batch_size,
    int physical_blocks,
    int max_blocks_per_seq,
    int query_heads,
    int kv_heads,
    int block_size,
    int head_dim,
    int capacity,
    int splits,
    int blocks_per_split) {
  constexpr int kTokenTile = 16;
  constexpr int kGroupSize = 16;
  constexpr int kWarpsPerCTA = 4;
  constexpr int kHeadsPerWarp = kGroupSize / kWarpsPerCTA;
  constexpr int kElementsPerVector = sizeof(uint4) / sizeof(T);
  static_assert(sizeof(T) == 2,
                "Tensor-core Stage 2 requires a 16-bit cache dtype.");
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_id = blockIdx.z;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (token_id >= tokens || kv_head >= kv_heads || split_id >= splits) {
    return;
  }
  const int first_query_head = kv_head * kGroupSize + warp_id * kHeadsPerWarp;
  const int values_per_lane = (head_dim + 31) / 32;

  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    for (int head = 0; head < kHeadsPerWarp; ++head) {
      const int query_head = first_query_head + head;
      const int64_t scalar_offset =
          (static_cast<int64_t>(token_id) * query_heads + query_head) * splits +
          split_id;
      const int64_t acc_offset = scalar_offset * head_dim;
      for (int element = 0; element < values_per_lane; ++element) {
        const int dim = lane + element * 32;
        if (dim < head_dim) {
          partial_acc[acc_offset + dim] = 0.0f;
        }
      }
      if (lane == 0) {
        partial_max[scalar_offset] = -CUDART_INF_F;
        partial_sum[scalar_offset] = 0.0f;
      }
    }
    return;
  }

  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int* selected =
      topk_indices + (token_id * kv_heads + kv_head) * capacity;
  const int first_slot = split_id * blocks_per_split;
  const int last_slot = min(capacity, first_slot + blocks_per_split);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  extern __shared__ char tensor_attention_shared[];
  T* shared_key = reinterpret_cast<T*>(tensor_attention_shared);
  T* shared_value = shared_key + kTokenTile * head_dim;
  const size_t scores_offset =
      (static_cast<size_t>(2 * kTokenTile * head_dim) * sizeof(T) +
       alignof(float) - 1) &
      ~(alignof(float) - 1);
  float* shared_scores =
      reinterpret_cast<float*>(tensor_attention_shared + scores_offset);
  const size_t probability_offset =
      scores_offset + kGroupSize * kTokenTile * sizeof(float);
  T* shared_probability =
      reinterpret_cast<T*>(tensor_attention_shared + probability_offset);
  const size_t output_offset =
      probability_offset + kGroupSize * kTokenTile * sizeof(T);
  float* shared_output =
      reinterpret_cast<float*>(tensor_attention_shared + output_offset);
  float* shared_pv = shared_output + kGroupSize * head_dim;
  float* shared_row_max = shared_pv + kWarpsPerCTA * kGroupSize * kTokenTile;
  float* shared_row_sum = shared_row_max + kGroupSize;
  float* shared_old_scale = shared_row_sum + kGroupSize;

  for (int index = threadIdx.x; index < kGroupSize * head_dim;
       index += blockDim.x) {
    shared_output[index] = 0.0f;
  }
  if (threadIdx.x < kGroupSize) {
    shared_row_max[threadIdx.x] = -CUDART_INF_F;
    shared_row_sum[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int slot = first_slot; slot < last_slot; ++slot) {
    const int logical_block = selected[slot];
    if (logical_block < 0) {
      break;
    }
    const int physical_block =
        block_tables[batch_id * max_blocks_per_seq + logical_block];
    if (physical_block < 0 || physical_block >= physical_blocks) {
      continue;
    }
    for (int block_offset = 0; block_offset < block_size;
         block_offset += kTokenTile) {
      const int first_position = logical_block * block_size + block_offset;
      if (first_position > position) {
        break;
      }
      const int tile_tokens =
          min(kTokenTile,
              min(block_size - block_offset, position - first_position + 1));
      const int64_t cache_tile_offset =
          ((static_cast<int64_t>(physical_block) * kv_heads + kv_head) *
               block_size +
           block_offset) *
          head_dim;
      if (tile_tokens == kTokenTile) {
        const int tile_vectors = kTokenTile * head_dim / kElementsPerVector;
        for (int vector_index = threadIdx.x; vector_index < tile_vectors;
             vector_index += blockDim.x) {
          const int element_index = vector_index * kElementsPerVector;
          const uint4 packed_key = *reinterpret_cast<const uint4*>(
              key_cache + cache_tile_offset + element_index);
          const uint4 packed_value = *reinterpret_cast<const uint4*>(
              value_cache + cache_tile_offset + element_index);
          *reinterpret_cast<uint4*>(shared_key + element_index) = packed_key;
          *reinterpret_cast<uint4*>(shared_value + element_index) =
              packed_value;
        }
      } else {
        for (int index = threadIdx.x; index < kTokenTile * head_dim;
             index += blockDim.x) {
          const int tile_token = index / head_dim;
          if (tile_token < tile_tokens) {
            shared_key[index] = key_cache[cache_tile_offset + index];
            shared_value[index] = value_cache[cache_tile_offset + index];
          } else {
            shared_key[index] = InfLLMFromFloat<T>(0.0f);
            shared_value[index] = InfLLMFromFloat<T>(0.0f);
          }
        }
      }
      __syncthreads();

      if (warp_id == 0) {
        using namespace nvcuda;
        wmma::fragment<wmma::accumulator,
                       kTokenTile,
                       kTokenTile,
                       kTokenTile,
                       float>
            score_fragment;
        wmma::fill_fragment(score_fragment, 0.0f);
        const T* query_base =
            query + (static_cast<int64_t>(token_id) * query_heads +
                     kv_head * kGroupSize) *
                        head_dim;
        for (int dim = 0; dim < head_dim; dim += kTokenTile) {
          wmma::fragment<wmma::matrix_a,
                         kTokenTile,
                         kTokenTile,
                         kTokenTile,
                         T,
                         wmma::row_major>
              query_fragment;
          wmma::fragment<wmma::matrix_b,
                         kTokenTile,
                         kTokenTile,
                         kTokenTile,
                         T,
                         wmma::col_major>
              key_fragment;
          wmma::load_matrix_sync(query_fragment, query_base + dim, head_dim);
          wmma::load_matrix_sync(key_fragment, shared_key + dim, head_dim);
          wmma::mma_sync(
              score_fragment, query_fragment, key_fragment, score_fragment);
        }
        wmma::store_matrix_sync(
            shared_scores, score_fragment, kTokenTile, wmma::mem_row_major);
      }
      __syncthreads();

      if (threadIdx.x < kGroupSize) {
        const int group_head = threadIdx.x;
        const int score_base = group_head * kTokenTile;
        float tile_max = -CUDART_INF_F;
        for (int tile_token = 0; tile_token < tile_tokens; ++tile_token) {
          tile_max =
              fmaxf(tile_max, shared_scores[score_base + tile_token] * scale);
        }
        const float previous_max = shared_row_max[group_head];
        const float next_max = fmaxf(previous_max, tile_max);
        const float old_scale = previous_max == -CUDART_INF_F
                                    ? 0.0f
                                    : expf(previous_max - next_max);
        float tile_sum = 0.0f;
        for (int tile_token = 0; tile_token < kTokenTile; ++tile_token) {
          const float probability =
              tile_token < tile_tokens
                  ? expf(shared_scores[score_base + tile_token] * scale -
                         next_max)
                  : 0.0f;
          shared_probability[score_base + tile_token] =
              InfLLMFromFloat<T>(probability);
          tile_sum += probability;
        }
        shared_old_scale[group_head] = old_scale;
        shared_row_max[group_head] = next_max;
        shared_row_sum[group_head] =
            shared_row_sum[group_head] * old_scale + tile_sum;
      }
      __syncthreads();

      using namespace nvcuda;
      for (int dim_tile = warp_id; dim_tile < head_dim / kTokenTile;
           dim_tile += kWarpsPerCTA) {
        wmma::fragment<wmma::matrix_a,
                       kGroupSize,
                       kTokenTile,
                       kTokenTile,
                       T,
                       wmma::row_major>
            probability_fragment;
        wmma::fragment<wmma::matrix_b,
                       kGroupSize,
                       kTokenTile,
                       kTokenTile,
                       T,
                       wmma::row_major>
            value_fragment;
        wmma::fragment<wmma::accumulator,
                       kGroupSize,
                       kTokenTile,
                       kTokenTile,
                       float>
            output_fragment;
        wmma::fill_fragment(output_fragment, 0.0f);
        wmma::load_matrix_sync(
            probability_fragment, shared_probability, kTokenTile);
        wmma::load_matrix_sync(
            value_fragment, shared_value + dim_tile * kTokenTile, head_dim);
        wmma::mma_sync(output_fragment,
                       probability_fragment,
                       value_fragment,
                       output_fragment);
        float* warp_pv = shared_pv + warp_id * kGroupSize * kTokenTile;
        wmma::store_matrix_sync(
            warp_pv, output_fragment, kTokenTile, wmma::mem_row_major);
        __syncwarp();
        for (int index = lane; index < kGroupSize * kTokenTile; index += 32) {
          const int group_head = index / kTokenTile;
          const int dim = dim_tile * kTokenTile + index % kTokenTile;
          const int output_index = group_head * head_dim + dim;
          shared_output[output_index] =
              shared_output[output_index] * shared_old_scale[group_head] +
              warp_pv[index];
        }
        __syncwarp();
      }
      __syncthreads();
    }
  }

  for (int index = threadIdx.x; index < kGroupSize * head_dim;
       index += blockDim.x) {
    const int group_head = index / head_dim;
    const int dim = index % head_dim;
    const int query_head = kv_head * kGroupSize + group_head;
    const int64_t scalar_offset =
        (static_cast<int64_t>(token_id) * query_heads + query_head) * splits +
        split_id;
    partial_acc[scalar_offset * head_dim + dim] = shared_output[index];
  }
  if (threadIdx.x < kGroupSize) {
    const int query_head = kv_head * kGroupSize + threadIdx.x;
    const int64_t scalar_offset =
        (static_cast<int64_t>(token_id) * query_heads + query_head) * splits +
        split_id;
    partial_max[scalar_offset] = shared_row_max[threadIdx.x];
    partial_sum[scalar_offset] = shared_row_sum[threadIdx.x];
  }
}

template <typename T>
__global__ void InfLLMV2SparseAttentionSplitKVKernel(
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* block_tables,
    const int* seq_lens_decoder,
    const int* batch_id_per_token,
    const int* cu_seqlens_q,
    const int* topk_indices,
    float* partial_acc,
    float* partial_max,
    float* partial_sum,
    int tokens,
    int batch_size,
    int physical_blocks,
    int max_blocks_per_seq,
    int query_heads,
    int kv_heads,
    int block_size,
    int head_dim,
    int capacity,
    int splits,
    int blocks_per_split) {
  constexpr int kTokenTile = 8;
  const int token_id = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_id = blockIdx.z;
  const int group_size = query_heads / kv_heads;
  const int group_head = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (token_id >= tokens || kv_head >= kv_heads || split_id >= splits ||
      group_head >= group_size) {
    return;
  }
  const int query_head = kv_head * group_size + group_head;
  const int values_per_lane = (head_dim + 31) / 32;
  float query_values[8];
  float accumulator[8];
#pragma unroll
  for (int element = 0; element < 8; ++element) {
    query_values[element] = 0.0f;
    accumulator[element] = 0.0f;
  }
  const int64_t query_offset =
      (static_cast<int64_t>(token_id) * query_heads + query_head) * head_dim;
  for (int element = 0; element < values_per_lane; ++element) {
    const int dim = lane + element * 32;
    if (dim < head_dim) {
      query_values[element] = InfLLMToFloat(query[query_offset + dim]);
    }
  }

  const int64_t partial_scalar_offset =
      (static_cast<int64_t>(token_id) * query_heads + query_head) * splits +
      split_id;
  const int64_t partial_acc_offset = partial_scalar_offset * head_dim;
  const int batch_id = batch_id_per_token[token_id];
  if (batch_id < 0 || batch_id >= batch_size) {
    for (int element = 0; element < values_per_lane; ++element) {
      const int dim = lane + element * 32;
      if (dim < head_dim) {
        partial_acc[partial_acc_offset + dim] = 0.0f;
      }
    }
    if (lane == 0) {
      partial_max[partial_scalar_offset] = -CUDART_INF_F;
      partial_sum[partial_scalar_offset] = 0.0f;
    }
    return;
  }
  const int position =
      InfLLMQueryPosition(token_id, batch_id, seq_lens_decoder, cu_seqlens_q);
  const int* selected =
      topk_indices + (token_id * kv_heads + kv_head) * capacity;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int first_slot = split_id * blocks_per_split;
  const int last_slot = min(capacity, first_slot + blocks_per_split);

  extern __shared__ char shared_bytes[];
  T* shared_key = reinterpret_cast<T*>(shared_bytes);
  T* shared_value = shared_key + kTokenTile * head_dim;
  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;
  for (int slot = first_slot; slot < last_slot; ++slot) {
    const int logical_block = selected[slot];
    if (logical_block < 0) {
      break;
    }
    const int physical_block =
        block_tables[batch_id * max_blocks_per_seq + logical_block];
    if (physical_block < 0 || physical_block >= physical_blocks) {
      continue;
    }
    for (int block_offset = 0; block_offset < block_size;
         block_offset += kTokenTile) {
      const int first_position = logical_block * block_size + block_offset;
      if (first_position > position) {
        break;
      }
      const int tile_tokens =
          min(kTokenTile,
              min(block_size - block_offset, position - first_position + 1));
      for (int index = threadIdx.x; index < tile_tokens * head_dim;
           index += blockDim.x) {
        const int tile_token = index / head_dim;
        const int dim = index % head_dim;
        const int64_t cache_offset =
            ((static_cast<int64_t>(physical_block) * kv_heads + kv_head) *
                 block_size +
             block_offset + tile_token) *
                head_dim +
            dim;
        shared_key[index] = key_cache[cache_offset];
        shared_value[index] = value_cache[cache_offset];
      }
      __syncthreads();

      for (int tile_token = 0; tile_token < tile_tokens; ++tile_token) {
        const int tile_offset = tile_token * head_dim;
        float score = 0.0f;
        for (int element = 0; element < values_per_lane; ++element) {
          const int dim = lane + element * 32;
          if (dim < head_dim) {
            score += query_values[element] *
                     InfLLMToFloat(shared_key[tile_offset + dim]);
          }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
          score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score = __shfl_sync(0xffffffff, score, 0) * scale;
        float old_weight = 1.0f;
        float new_weight = 1.0f;
        if (score > row_max) {
          old_weight = expf(row_max - score);
          row_max = score;
        } else {
          new_weight = expf(score - row_max);
        }
        row_sum = row_sum * old_weight + new_weight;
        for (int element = 0; element < values_per_lane; ++element) {
          const int dim = lane + element * 32;
          if (dim < head_dim) {
            accumulator[element] =
                accumulator[element] * old_weight +
                new_weight * InfLLMToFloat(shared_value[tile_offset + dim]);
          }
        }
      }
      __syncthreads();
    }
  }
  for (int element = 0; element < values_per_lane; ++element) {
    const int dim = lane + element * 32;
    if (dim < head_dim) {
      partial_acc[partial_acc_offset + dim] = accumulator[element];
    }
  }
  if (lane == 0) {
    partial_max[partial_scalar_offset] = row_max;
    partial_sum[partial_scalar_offset] = row_sum;
  }
}

template <typename T>
__global__ void InfLLMV2SparseAttentionCombineKernel(const float* partial_acc,
                                                     const float* partial_max,
                                                     const float* partial_sum,
                                                     T* output,
                                                     int tokens,
                                                     int query_heads,
                                                     int head_dim,
                                                     int splits) {
  const int token_id = blockIdx.x;
  const int query_head = blockIdx.y;
  const int thread = threadIdx.x;
  if (token_id >= tokens || query_head >= query_heads) {
    return;
  }
  const int64_t scalar_offset =
      (static_cast<int64_t>(token_id) * query_heads + query_head) * splits;
  extern __shared__ float reduction[];
  float local_max = -CUDART_INF_F;
  for (int split = thread; split < splits; split += blockDim.x) {
    local_max = fmaxf(local_max, partial_max[scalar_offset + split]);
  }
  reduction[thread] = local_max;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (thread < width) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + width]);
    }
    __syncthreads();
  }
  const float row_max = reduction[0];
  float local_sum = 0.0f;
  if (row_max != -CUDART_INF_F) {
    for (int split = thread; split < splits; split += blockDim.x) {
      const float split_sum = partial_sum[scalar_offset + split];
      if (split_sum > 0.0f) {
        local_sum +=
            split_sum * expf(partial_max[scalar_offset + split] - row_max);
      }
    }
  }
  reduction[thread] = local_sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (thread < width) {
      reduction[thread] += reduction[thread + width];
    }
    __syncthreads();
  }
  const float row_sum = reduction[0];
  const int64_t output_offset =
      (static_cast<int64_t>(token_id) * query_heads + query_head) * head_dim;
  for (int dim = thread; dim < head_dim; dim += blockDim.x) {
    float value = 0.0f;
    if (row_sum > 0.0f) {
      for (int split = 0; split < splits; ++split) {
        const float split_sum = partial_sum[scalar_offset + split];
        if (split_sum > 0.0f) {
          const int64_t acc_offset = (scalar_offset + split) * head_dim + dim;
          value += partial_acc[acc_offset] *
                   expf(partial_max[scalar_offset + split] - row_max);
        }
      }
      value /= row_sum;
    }
    output[output_offset + dim] = InfLLMFromFloat<T>(value);
  }
}

}  // namespace fastdeploy
