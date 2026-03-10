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

// V100 (SM70) decode attention CUDA kernel.
// Replaces Triton flash-decoding kernels to eliminate torch_proxy launch
// overhead (~1.5ms per Triton kernel launch).
//
// Three kernels:
// 1. v100_write_kv_cache_kernel  - write new K/V to paged block cache
// 2. v100_decode_attn_stage1_kernel - flash-decoding with online softmax
// 3. v100_decode_attn_stage2_kernel - merge partial outputs across splits

#include "helper.h"

// ============================================================================
// Warp/Block reduce utilities (SM70 compatible)
// ============================================================================

__device__ __forceinline__ float warpReduceSum(float val) {
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  return val;
}

// blockReduceSum for 128 threads (4 warps).
// Uses 4 floats of shared memory for cross-warp communication.
__device__ __forceinline__ float blockReduceSum4(float val,
                                                 float* smem_scratch) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) smem_scratch[warp] = val;
  __syncthreads();

  // Warp 0 reduces across warps
  val = (threadIdx.x < 4) ? smem_scratch[threadIdx.x] : 0.f;
  if (warp == 0) val = warpReduceSum(val);

  // Broadcast result from thread 0 via shared memory
  if (threadIdx.x == 0) smem_scratch[0] = val;
  __syncthreads();
  return smem_scratch[0];
}

// ============================================================================
// Kernel 1: Write KV to block cache
// ============================================================================
// Grid: (num_tokens * kv_num_heads), Block: (HEAD_DIM) or (128) if HEAD_DIM>128
// Each thread block handles one (token, kv_head) pair.

template <typename T>
__global__ void v100_write_kv_cache_kernel(
    const T* __restrict__ k_new,  // [num_tokens, kv_num_heads, head_dim]
    const T* __restrict__ v_new,  // [num_tokens, kv_num_heads, head_dim]
    T* __restrict__ key_cache,    // [max_num_blocks, kv_num_heads, block_size,
                                  // head_dim]
    T* __restrict__ value_cache,  // same layout
    const int* __restrict__ block_tables,   // [batch_size, max_blocks_per_seq]
    const int64_t* __restrict__ positions,  // [num_tokens] int64
    const int* __restrict__ batch_ids,      // [num_tokens] int32
    const int num_tokens,
    const int kv_num_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq) {
  const int pid = blockIdx.x;
  const int token_id = pid / kv_num_heads;
  const int head_id = pid % kv_num_heads;

  if (token_id >= num_tokens) return;

  const int64_t pos = positions[token_id];
  const int batch_id = batch_ids[token_id];
  const int block_idx = static_cast<int>(pos / block_size);
  const int block_offset = static_cast<int>(pos % block_size);

  const int physical_block =
      __ldg(&block_tables[batch_id * max_blocks_per_seq + block_idx]);

  // Source offset: k_new[token_id, head_id, :]
  const int64_t src_base =
      static_cast<int64_t>(token_id) * kv_num_heads * head_dim +
      head_id * head_dim;
  // Dest offset: cache[physical_block, head_id, block_offset, :]
  const int64_t dst_base = static_cast<int64_t>(physical_block) * kv_num_heads *
                               block_size * head_dim +
                           head_id * block_size * head_dim +
                           block_offset * head_dim;

  // Vectorized copy using float4 (8 half values at once)
  const int vec_size = 8;  // sizeof(float4) / sizeof(half) = 8
  const int num_vecs = head_dim / vec_size;

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const int offset = i * vec_size;
    // Load from k_new/v_new
    float4 k_val = *reinterpret_cast<const float4*>(&k_new[src_base + offset]);
    float4 v_val = *reinterpret_cast<const float4*>(&v_new[src_base + offset]);
    // Store to cache
    *reinterpret_cast<float4*>(&key_cache[dst_base + offset]) = k_val;
    *reinterpret_cast<float4*>(&value_cache[dst_base + offset]) = v_val;
  }

  // Handle remainder if head_dim is not divisible by vec_size
  const int remainder_start = num_vecs * vec_size;
  for (int i = remainder_start + threadIdx.x; i < head_dim; i += blockDim.x) {
    key_cache[dst_base + i] = k_new[src_base + i];
    value_cache[dst_base + i] = v_new[src_base + i];
  }
}

// ============================================================================
// Kernel 2: Decode attention stage 1 (flash-decoding with online softmax)
// ============================================================================
// Grid: (batch_size, num_heads, num_kv_splits), Block: (THREADS)
// Each thread block computes attention for one (batch, q_head, split).
// THREADS should be >= HEAD_DIM for full utilization.
// Each thread handles HEAD_DIM/THREADS elements of the head dimension.

template <typename T, bool SINGLE_SPLIT>
__global__ void v100_decode_attn_stage1_kernel(
    const T* __restrict__ q,               // [num_tokens, num_heads, head_dim]
    const T* __restrict__ key_cache,       // [max_num_blocks, kv_num_heads,
                                           // block_size, head_dim]
    const T* __restrict__ value_cache,     // same layout
    T* __restrict__ output,                // [num_tokens, num_heads, head_dim]
    float* __restrict__ partial_out,       // [batch, num_heads, num_kv_splits,
                                           // head_dim]
    float* __restrict__ partial_lse,       // [batch, num_heads, num_kv_splits]
    const int* __restrict__ block_tables,  // [batch_size, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [batch_size] int32
    const int* __restrict__ q_start_locs,  // [batch_size] int32
    const float sm_scale,
    const int max_blocks_per_seq,
    const int num_heads,
    const int kv_num_heads,
    const int group_size,
    const int head_dim,
    const int block_size,
    const int num_kv_splits,
    const int max_blocks_per_split) {
  const int pid_batch = blockIdx.x;
  const int pid_head = blockIdx.y;
  const int pid_split = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // Shared memory for cross-warp reduce (4 warps → 4 floats)
  __shared__ float smem_scratch[WARP_SIZE];

  const int total_kv_len = __ldg(&seq_lens[pid_batch]);
  if (total_kv_len <= 0) {
    if (!SINGLE_SPLIT) {
      // Write sentinel values so stage2 knows this split is empty
      for (int d = tid; d < head_dim; d += num_threads) {
        const int64_t out_idx = static_cast<int64_t>(pid_batch) * num_heads *
                                    num_kv_splits * head_dim +
                                pid_head * num_kv_splits * head_dim +
                                pid_split * head_dim + d;
        partial_out[out_idx] = 0.f;
      }
      if (tid == 0) {
        const int64_t lse_idx =
            static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits +
            pid_head * num_kv_splits + pid_split;
        partial_lse[lse_idx] = -INFINITY;
      }
    }
    return;
  }

  const int kv_head_id = pid_head / group_size;

  // Determine block range for this split
  const int total_kv_blocks = (total_kv_len + block_size - 1) / block_size;
  const int blocks_per_split =
      (total_kv_blocks + num_kv_splits - 1) / num_kv_splits;
  const int split_start = pid_split * blocks_per_split;
  int split_end = min((pid_split + 1) * blocks_per_split, total_kv_blocks);

  if (split_start >= total_kv_blocks) {
    if (!SINGLE_SPLIT) {
      for (int d = tid; d < head_dim; d += num_threads) {
        const int64_t out_idx = static_cast<int64_t>(pid_batch) * num_heads *
                                    num_kv_splits * head_dim +
                                pid_head * num_kv_splits * head_dim +
                                pid_split * head_dim + d;
        partial_out[out_idx] = 0.f;
      }
      if (tid == 0) {
        const int64_t lse_idx =
            static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits +
            pid_head * num_kv_splits + pid_split;
        partial_lse[lse_idx] = -INFINITY;
      }
    }
    return;
  }

  // Load Q vector: each thread loads elements it's responsible for
  const int q_start = __ldg(&q_start_locs[pid_batch]);
  const int64_t q_base = static_cast<int64_t>(q_start) * num_heads * head_dim +
                         pid_head * head_dim;

  // Number of elements per thread (handle HEAD_DIM > num_threads)
  const int elems_per_thread = (head_dim + num_threads - 1) / num_threads;

  // Register storage for Q, accumulator
  // Max 4 elements per thread (supports HEAD_DIM up to 512 with 128 threads)
  float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
  float acc_reg[4] = {0.f, 0.f, 0.f, 0.f};

  for (int e = 0; e < elems_per_thread; e++) {
    const int d = tid + e * num_threads;
    if (d < head_dim) {
      q_reg[e] = static_cast<float>(q[q_base + d]);
    }
  }

  // Online softmax state
  float m_i = -INFINITY;
  float l_i = 0.f;

  // Cache base addresses
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * head_dim;
  const int64_t kv_block_stride =
      static_cast<int64_t>(kv_num_heads) * kv_head_stride;

  // Iterate over KV blocks in this split
  for (int bi = split_start; bi < split_end; bi++) {
    const int physical_block =
        __ldg(&block_tables[pid_batch * max_blocks_per_seq + bi]);
    const int block_start_pos = bi * block_size;
    const int valid_tokens = min(block_size, total_kv_len - block_start_pos);

    const int64_t cache_block_base =
        static_cast<int64_t>(physical_block) * kv_block_stride +
        kv_head_id * kv_head_stride;

    // Process each KV token in this block
    for (int kv = 0; kv < valid_tokens; kv++) {
      const int64_t kv_offset = cache_block_base + kv * head_dim;

      // Compute dot product: Q . K
      float qk_local = 0.f;
      for (int e = 0; e < elems_per_thread; e++) {
        const int d = tid + e * num_threads;
        if (d < head_dim) {
          float k_val = static_cast<float>(key_cache[kv_offset + d]);
          qk_local += q_reg[e] * k_val;
        }
      }

      // Block-wide reduce to get full dot product
      float qk = blockReduceSum4(qk_local, smem_scratch);
      qk *= sm_scale;

      // Online softmax update (all threads see the same qk after reduce)
      float m_new = fmaxf(m_i, qk);
      float alpha = __expf(m_i - m_new);
      float p = __expf(qk - m_new);
      l_i = l_i * alpha + p;

      // Load V and update accumulator
      for (int e = 0; e < elems_per_thread; e++) {
        const int d = tid + e * num_threads;
        if (d < head_dim) {
          float v_val = static_cast<float>(value_cache[kv_offset + d]);
          acc_reg[e] = acc_reg[e] * alpha + p * v_val;
        }
      }

      m_i = m_new;
    }
  }

  // Write results
  if (SINGLE_SPLIT) {
    // Direct output write
    const int64_t out_base =
        static_cast<int64_t>(q_start) * num_heads * head_dim +
        pid_head * head_dim;
    float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * num_threads;
      if (d < head_dim) {
        output[out_base + d] = static_cast<T>(acc_reg[e] * inv_l);
      }
    }
  } else {
    // Write partial output + LSE for stage2
    const int64_t partial_base =
        static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits * head_dim +
        pid_head * num_kv_splits * head_dim + pid_split * head_dim;
    float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * num_threads;
      if (d < head_dim) {
        partial_out[partial_base + d] = acc_reg[e] * inv_l;
      }
    }
    if (tid == 0) {
      const int64_t lse_idx =
          static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits +
          pid_head * num_kv_splits + pid_split;
      partial_lse[lse_idx] = m_i + logf(l_i);
    }
  }
}

// ============================================================================
// Kernel 3: Decode attention stage 2 (merge partial outputs)
// ============================================================================
// Grid: (batch_size, num_heads), Block: (THREADS)
// Merges num_kv_splits partial outputs using LSE-based rescaling.

template <typename T>
__global__ void v100_decode_attn_stage2_kernel(
    const float* __restrict__ partial_out,  // [batch, heads, splits, head_dim]
    const float* __restrict__ partial_lse,  // [batch, heads, splits]
    T* __restrict__ output,                 // [num_tokens, heads, head_dim]
    const int* __restrict__ q_start_locs,   // [batch] int32
    const int* __restrict__ seq_lens,       // [batch] int32
    const int num_heads,
    const int head_dim,
    const int num_kv_splits) {
  const int pid_batch = blockIdx.x;
  const int pid_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int total_kv_len = __ldg(&seq_lens[pid_batch]);
  if (total_kv_len <= 0) return;

  const int elems_per_thread = (head_dim + num_threads - 1) / num_threads;

  // Find max LSE across splits
  float max_lse = -INFINITY;
  const int64_t lse_base =
      static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits +
      pid_head * num_kv_splits;

  for (int s = 0; s < num_kv_splits; s++) {
    float lse_val = partial_lse[lse_base + s];
    max_lse = fmaxf(max_lse, lse_val);
  }

  // Merge: weighted sum with LSE rescaling
  float sum_exp = 0.f;
  float acc_reg[4] = {0.f, 0.f, 0.f, 0.f};

  const int64_t partial_head_base =
      static_cast<int64_t>(pid_batch) * num_heads * num_kv_splits * head_dim +
      pid_head * num_kv_splits * head_dim;

  for (int s = 0; s < num_kv_splits; s++) {
    float lse_val = partial_lse[lse_base + s];
    bool is_valid = (lse_val > -INFINITY);
    float w = is_valid ? __expf(lse_val - max_lse) : 0.f;
    sum_exp += w;

    const int64_t split_base = partial_head_base + s * head_dim;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * num_threads;
      if (d < head_dim) {
        float pval = is_valid ? partial_out[split_base + d] : 0.f;
        acc_reg[e] += w * pval;
      }
    }
  }

  // Normalize and write output
  const int q_start = __ldg(&q_start_locs[pid_batch]);
  const int64_t out_base =
      static_cast<int64_t>(q_start) * num_heads * head_dim +
      pid_head * head_dim;
  float inv_sum = (sum_exp > 0.f) ? (1.f / sum_exp) : 0.f;

  for (int e = 0; e < elems_per_thread; e++) {
    const int d = tid + e * num_threads;
    if (d < head_dim) {
      output[out_base + d] = static_cast<T>(acc_reg[e] * inv_sum);
    }
  }
}

// ============================================================================
// Host wrapper function
// ============================================================================

void V100DecodeAttention(
    paddle::Tensor& output,       // [num_tokens, num_heads, head_dim]
    const paddle::Tensor& q,      // [num_tokens, num_heads, head_dim]
    const paddle::Tensor& k_new,  // [num_tokens, kv_num_heads, head_dim]
    const paddle::Tensor& v_new,  // [num_tokens, kv_num_heads, head_dim]
    paddle::Tensor&
        key_cache,  // [max_num_blocks, kv_num_heads, block_size, head_dim]
    paddle::Tensor& value_cache,         // same layout
    const paddle::Tensor& block_tables,  // [batch_size, max_blocks_per_seq]
    const paddle::Tensor& seq_lens,      // [batch_size] int32
    const paddle::Tensor& positions,     // [num_tokens] int64
    const paddle::Tensor& batch_ids,     // [num_tokens] int32
    const paddle::Tensor& q_start_locs,  // [batch_size] int32
    float sm_scale,
    int num_kv_splits,
    int max_blocks_per_split,
    bool skip_kv_write) {
  auto stream = q.stream();

  const int num_tokens = q.dims()[0];
  const int num_heads = q.dims()[1];
  const int head_dim = q.dims()[2];
  const int kv_num_heads = k_new.dims()[1];
  const int block_size = key_cache.dims()[2];
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int batch_size = seq_lens.dims()[0];
  const int group_size = num_heads / kv_num_heads;
  const bool single_split = (num_kv_splits == 1);

  const int THREADS = 128;

  PD_CHECK(head_dim <= THREADS * 4,
           "V100 decode attention supports head_dim up to ",
           THREADS * 4,
           " but got ",
           head_dim);

  // ---- Kernel 1: Write KV to cache (skip if already written by
  // v100_rope_write_cache) ----
  if (!skip_kv_write) {
    const int grid_size = num_tokens * kv_num_heads;
    const int block_threads = min(head_dim, THREADS);
    dim3 grid(grid_size);
    dim3 block(block_threads);

    PD_DISPATCH_FLOATING_AND_HALF_TYPES(
        q.dtype(), "v100_write_kv_cache_kernel", [&] {
          v100_write_kv_cache_kernel<data_t>
              <<<grid, block, 0, stream>>>(k_new.data<data_t>(),
                                           v_new.data<data_t>(),
                                           key_cache.data<data_t>(),
                                           value_cache.data<data_t>(),
                                           block_tables.data<int>(),
                                           positions.data<int64_t>(),
                                           batch_ids.data<int>(),
                                           num_tokens,
                                           kv_num_heads,
                                           head_dim,
                                           block_size,
                                           max_blocks_per_seq);
        });
  }

  // ---- Kernel 2: Decode attention ----
  // Shared memory: 4 floats for cross-warp reduce scratch
  const int smem_size = WARP_SIZE * sizeof(float);

  if (single_split) {
    dim3 grid(batch_size, num_heads, 1);
    dim3 block(THREADS);

    PD_DISPATCH_FLOATING_AND_HALF_TYPES(
        q.dtype(), "v100_decode_attn_stage1_single", [&] {
          v100_decode_attn_stage1_kernel<data_t, true>
              <<<grid, block, smem_size, stream>>>(
                  q.data<data_t>(),
                  key_cache.data<data_t>(),
                  value_cache.data<data_t>(),
                  output.data<data_t>(),
                  nullptr,  // partial_out unused
                  nullptr,  // partial_lse unused
                  block_tables.data<int>(),
                  seq_lens.data<int>(),
                  q_start_locs.data<int>(),
                  sm_scale,
                  max_blocks_per_seq,
                  num_heads,
                  kv_num_heads,
                  group_size,
                  head_dim,
                  block_size,
                  1,  // num_kv_splits
                  max_blocks_per_split);
        });
  } else {
    // Allocate partial buffers
    auto partial_out =
        GetEmptyTensor({batch_size, num_heads, num_kv_splits, head_dim},
                       paddle::DataType::FLOAT32,
                       q.place());
    auto partial_lse = GetEmptyTensor({batch_size, num_heads, num_kv_splits},
                                      paddle::DataType::FLOAT32,
                                      q.place());

    dim3 grid(batch_size, num_heads, num_kv_splits);
    dim3 block(THREADS);

    PD_DISPATCH_FLOATING_AND_HALF_TYPES(
        q.dtype(), "v100_decode_attn_stage1_multi", [&] {
          v100_decode_attn_stage1_kernel<data_t, false>
              <<<grid, block, smem_size, stream>>>(q.data<data_t>(),
                                                   key_cache.data<data_t>(),
                                                   value_cache.data<data_t>(),
                                                   output.data<data_t>(),
                                                   partial_out.data<float>(),
                                                   partial_lse.data<float>(),
                                                   block_tables.data<int>(),
                                                   seq_lens.data<int>(),
                                                   q_start_locs.data<int>(),
                                                   sm_scale,
                                                   max_blocks_per_seq,
                                                   num_heads,
                                                   kv_num_heads,
                                                   group_size,
                                                   head_dim,
                                                   block_size,
                                                   num_kv_splits,
                                                   max_blocks_per_split);
        });

    // ---- Kernel 3: Stage 2 merge ----
    {
      dim3 grid_s2(batch_size, num_heads);
      dim3 block_s2(THREADS);

      PD_DISPATCH_FLOATING_AND_HALF_TYPES(
          q.dtype(), "v100_decode_attn_stage2", [&] {
            v100_decode_attn_stage2_kernel<data_t>
                <<<grid_s2, block_s2, 0, stream>>>(partial_out.data<float>(),
                                                   partial_lse.data<float>(),
                                                   output.data<data_t>(),
                                                   q_start_locs.data<int>(),
                                                   seq_lens.data<int>(),
                                                   num_heads,
                                                   head_dim,
                                                   num_kv_splits);
          });
    }
  }
}

// ============================================================================
// PD_BUILD_STATIC_OP registration
// ============================================================================

PD_BUILD_STATIC_OP(v100_decode_attention)
    .Inputs({"output",
             "q",
             "k_new",
             "v_new",
             "key_cache",
             "value_cache",
             "block_tables",
             "seq_lens",
             "positions",
             "batch_ids",
             "q_start_locs"})
    .Outputs({"output_out", "key_cache_out", "value_cache_out"})
    .Attrs({"sm_scale: float",
            "num_kv_splits: int",
            "max_blocks_per_split: int",
            "skip_kv_write: bool"})
    .SetInplaceMap({{"output", "output_out"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(V100DecodeAttention));
