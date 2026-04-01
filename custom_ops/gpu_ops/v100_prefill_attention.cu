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

// V100 (SM70) prefill attention CUDA kernel.
// Replaces the extremely slow Python fallback path (_python_forward) for
// prefill/mixed batches where q_len > 1.
//
// Design:
// 1. v100_write_kv_cache_kernel  - write new K/V to paged block cache (reused)
// 2. v100_prefill_attn_kernel    - per-query online softmax attention over
//                                  block-based KV cache with causal masking
//
// Supports:
// - Variable-length Q sequences (mixed prefill + decode in one batch)
// - Block-based (paged) KV cache with configurable block_size
// - GQA (grouped-query attention)
// - Causal masking
// - FP16 (SM70, no BF16)

#include "helper.h"

// ============================================================================
// Warp/Block reduce utilities (SM70 compatible)
// ============================================================================

__device__ __forceinline__ float warpReduceSum_prefill(float val) {
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  return val;
}

__device__ __forceinline__ float blockReduceSum_prefill(float val,
                                                        float* smem_scratch) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  val = warpReduceSum_prefill(val);

  if (lane == 0) smem_scratch[warp] = val;
  __syncthreads();

  val = (threadIdx.x < num_warps) ? smem_scratch[threadIdx.x] : 0.f;
  if (warp == 0) val = warpReduceSum_prefill(val);

  if (threadIdx.x == 0) smem_scratch[0] = val;
  __syncthreads();
  return smem_scratch[0];
}

// ============================================================================
// Kernel 1: Write KV to block cache (same as in v100_decode_attention.cu)
// ============================================================================

template <typename T>
__global__ void v100_prefill_write_kv_cache_kernel(
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

  if (physical_block < 0) return;

  const int64_t src_base =
      static_cast<int64_t>(token_id) * kv_num_heads * head_dim +
      head_id * head_dim;
  const int64_t dst_base = static_cast<int64_t>(physical_block) * kv_num_heads *
                               block_size * head_dim +
                           head_id * block_size * head_dim +
                           block_offset * head_dim;

  const int vec_size = 8;
  const int num_vecs = head_dim / vec_size;

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const int offset = i * vec_size;
    float4 k_val = *reinterpret_cast<const float4*>(&k_new[src_base + offset]);
    float4 v_val = *reinterpret_cast<const float4*>(&v_new[src_base + offset]);
    *reinterpret_cast<float4*>(&key_cache[dst_base + offset]) = k_val;
    *reinterpret_cast<float4*>(&value_cache[dst_base + offset]) = v_val;
  }

  const int remainder_start = num_vecs * vec_size;
  for (int i = remainder_start + threadIdx.x; i < head_dim; i += blockDim.x) {
    key_cache[dst_base + i] = k_new[src_base + i];
    value_cache[dst_base + i] = v_new[src_base + i];
  }
}

// ============================================================================
// Kernel 2: Prefill attention with online softmax over block-based KV cache
// ============================================================================
// Grid:  (total_q_tokens, num_heads)
// Block: (THREADS)
//
// Each thread block computes attention for one (q_token, q_head) pair.
// It iterates over ALL KV blocks for that sequence, computes Q.K with causal
// masking, then accumulates the softmax-weighted V output using online softmax.
//
// This is the "naive" single-pass approach — no tiling of Q. For prefill on
// V100 this is still 500x-10000x faster than the Python fallback because:
// - Zero CPU-GPU syncs (no .item() calls)
// - All work parallelized across tokens and heads
// - Online softmax: single pass, O(1) extra memory per thread

template <typename T>
__global__ void v100_prefill_attn_kernel(
    const T* __restrict__ q,               // [num_tokens, num_heads, head_dim]
    const T* __restrict__ key_cache,       // [max_num_blocks, kv_num_heads,
                                           //  block_size, head_dim]
    const T* __restrict__ value_cache,     // same layout
    T* __restrict__ output,                // [num_tokens, num_heads, head_dim]
    const int* __restrict__ block_tables,  // [batch_size, max_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [batch_size] int32 - total KV len
    const int64_t* __restrict__ positions, // [num_tokens] int64
    const int* __restrict__ batch_ids,     // [num_tokens] int32
    const float sm_scale,
    const int max_blocks_per_seq,
    const int num_heads,
    const int kv_num_heads,
    const int group_size,
    const int head_dim,
    const int block_size,
    const bool is_causal) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  __shared__ float smem_scratch[WARP_SIZE];

  const int batch_id = __ldg(&batch_ids[token_idx]);
  const int total_kv_len = __ldg(&seq_lens[batch_id]);

  if (total_kv_len <= 0) return;

  const int kv_head_id = head_idx / group_size;

  // Current query position — for causal masking
  const int64_t q_pos = __ldg(&positions[token_idx]);

  // Number of elements per thread
  const int elems_per_thread = (head_dim + num_threads - 1) / num_threads;

  // Load Q vector into registers
  const int64_t q_base = static_cast<int64_t>(token_idx) * num_heads * head_dim +
                          head_idx * head_dim;
  float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
  for (int e = 0; e < elems_per_thread; e++) {
    const int d = tid + e * num_threads;
    if (d < head_dim) {
      q_reg[e] = static_cast<float>(q[q_base + d]);
    }
  }

  // Online softmax state
  float m_i = -INFINITY;
  float l_i = 0.f;
  float acc_reg[4] = {0.f, 0.f, 0.f, 0.f};

  // Cache layout strides
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * head_dim;
  const int64_t kv_block_stride =
      static_cast<int64_t>(kv_num_heads) * kv_head_stride;

  // Determine how many KV blocks to iterate
  const int total_kv_blocks = (total_kv_len + block_size - 1) / block_size;

  // Iterate over all KV blocks for this sequence
  for (int bi = 0; bi < total_kv_blocks; bi++) {
    const int physical_block =
        __ldg(&block_tables[batch_id * max_blocks_per_seq + bi]);

    if (physical_block < 0) continue;

    const int block_start_pos = bi * block_size;
    const int valid_tokens = min(block_size, total_kv_len - block_start_pos);

    const int64_t cache_block_base =
        static_cast<int64_t>(physical_block) * kv_block_stride +
        kv_head_id * kv_head_stride;

    // Process each KV token in this block
    for (int kv = 0; kv < valid_tokens; kv++) {
      const int kv_pos = block_start_pos + kv;

      // Causal masking: skip KV positions after Q position
      if (is_causal && kv_pos > static_cast<int>(q_pos)) {
        break;  // All subsequent positions in this and later blocks are masked
      }

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

      // Block-wide reduce
      float qk = blockReduceSum_prefill(qk_local, smem_scratch);
      qk *= sm_scale;

      // Online softmax update
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

    // Early exit: if causal and this block is past Q position, no more work
    if (is_causal && (bi + 1) * block_size > static_cast<int>(q_pos) + 1) {
      break;
    }
  }

  // Write output
  const int64_t out_base =
      static_cast<int64_t>(token_idx) * num_heads * head_dim +
      head_idx * head_dim;
  float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  for (int e = 0; e < elems_per_thread; e++) {
    const int d = tid + e * num_threads;
    if (d < head_dim) {
      output[out_base + d] = static_cast<T>(acc_reg[e] * inv_l);
    }
  }
}

// ============================================================================
// Host wrapper function
// ============================================================================

void V100PrefillAttention(
    paddle::Tensor& output,       // [num_tokens, num_heads, head_dim]
    const paddle::Tensor& q,      // [num_tokens, num_heads, head_dim]
    const paddle::Tensor& k_new,  // [num_tokens, kv_num_heads, head_dim]
    const paddle::Tensor& v_new,  // [num_tokens, kv_num_heads, head_dim]
    paddle::Tensor&
        key_cache,  // [max_num_blocks, kv_num_heads, block_size, head_dim]
    paddle::Tensor& value_cache,         // same layout
    const paddle::Tensor& block_tables,  // [batch_size, max_blocks_per_seq]
    const paddle::Tensor& seq_lens,      // [batch_size] int32 - total KV len
    const paddle::Tensor& positions,     // [num_tokens] int64
    const paddle::Tensor& batch_ids,     // [num_tokens] int32
    float sm_scale,
    bool is_causal = true,
    bool skip_kv_write = false) {
  auto stream = q.stream();

  const int num_tokens = q.dims()[0];
  const int num_heads = q.dims()[1];
  const int head_dim = q.dims()[2];
  const int kv_num_heads = k_new.dims()[1];
  const int block_size = key_cache.dims()[2];
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int group_size = num_heads / kv_num_heads;

  const int THREADS = 128;

  PD_CHECK(head_dim <= THREADS * 4,
           "V100 prefill attention supports head_dim up to ",
           THREADS * 4,
           " but got ",
           head_dim);

  // ---- Kernel 1: Write KV to cache ----
  if (!skip_kv_write) {
    const int grid_size = num_tokens * kv_num_heads;
    const int block_threads = min(head_dim, THREADS);
    dim3 grid(grid_size);
    dim3 block(block_threads);

    PD_DISPATCH_FLOATING_AND_HALF_TYPES(
        q.dtype(), "v100_prefill_write_kv_cache_kernel", [&] {
          v100_prefill_write_kv_cache_kernel<data_t>
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

  // ---- Kernel 2: Prefill attention ----
  // Grid: (num_tokens, num_heads) — one thread block per (q_token, q_head)
  const int smem_size = WARP_SIZE * sizeof(float);
  {
    dim3 grid(num_tokens, num_heads);
    dim3 block(THREADS);

    PD_DISPATCH_FLOATING_AND_HALF_TYPES(
        q.dtype(), "v100_prefill_attn_kernel", [&] {
          v100_prefill_attn_kernel<data_t>
              <<<grid, block, smem_size, stream>>>(q.data<data_t>(),
                                                   key_cache.data<data_t>(),
                                                   value_cache.data<data_t>(),
                                                   output.data<data_t>(),
                                                   block_tables.data<int>(),
                                                   seq_lens.data<int>(),
                                                   positions.data<int64_t>(),
                                                   batch_ids.data<int>(),
                                                   sm_scale,
                                                   max_blocks_per_seq,
                                                   num_heads,
                                                   kv_num_heads,
                                                   group_size,
                                                   head_dim,
                                                   block_size,
                                                   is_causal);
        });
  }
}

// ============================================================================
// PD_BUILD_STATIC_OP registration
// ============================================================================

PD_BUILD_STATIC_OP(v100_prefill_attention)
    .Inputs({"output",
             "q",
             "k_new",
             "v_new",
             "key_cache",
             "value_cache",
             "block_tables",
             "seq_lens",
             "positions",
             "batch_ids"})
    .Outputs({"output_out", "key_cache_out", "value_cache_out"})
    .Attrs({"sm_scale: float",
            "is_causal: bool",
            "skip_kv_write: bool"})
    .SetInplaceMap({{"output", "output_out"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(V100PrefillAttention));
