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

// V100 (SM70) compatible fused RoPE + KV cache write kernel.
// Does NOT use cp.async (requires SM80+), uses standard global memory access.
//
// Fuses two operations into a single kernel launch:
// 1. Apply NeoX-style RoPE to Q and K
// 2. Write K (after RoPE) and V to paged block cache
//
// Replaces Python implementations:
// - _python_apply_rope_to_qk()
// - _python_write_kv_to_block_cache()
//
// Grid: dim3(num_tokens, num_heads + kv_num_heads)
//   blockIdx.y < num_heads       : Q RoPE
//   blockIdx.y >= num_heads      : K RoPE + KV cache write
// Block: dim3(128)

#include "helper.h"
#include "paddle/extension.h"

template <typename T>
__global__ void v100_fused_rope_write_cache_kernel(
    const T* __restrict__ q_in,         // [num_tokens, num_heads, head_dim]
    const T* __restrict__ k_in,         // [num_tokens, kv_num_heads, head_dim]
    const T* __restrict__ v_in,         // [num_tokens, kv_num_heads, head_dim]
    const float* __restrict__ cos_emb,  // [max_seq_len, rotary_dim]
    const float* __restrict__ sin_emb,  // [max_seq_len, rotary_dim]
    T* __restrict__ q_out,              // [num_tokens, num_heads, head_dim]
    T* __restrict__ k_out,              // [num_tokens, kv_num_heads, head_dim]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_dim]
    T* __restrict__ value_cache,  // same layout
    const int* __restrict__ block_tables,   // [batch_size, max_blocks_per_seq]
    const int64_t* __restrict__ positions,  // [num_tokens]
    const int* __restrict__ batch_ids,      // [num_tokens]
    const int num_tokens,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int rotary_dim,
    const int block_size,
    const int max_blocks_per_seq) {
  const int token_id = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (token_id >= num_tokens) return;

  const int64_t pos = positions[token_id];
  const int half_head_dim = head_dim / 2;

  if (head_idx < num_heads) {
    // =====================================================================
    // Q: Apply NeoX RoPE only
    // =====================================================================
    const int64_t q_base =
        static_cast<int64_t>(token_id) * num_heads * head_dim +
        head_idx * head_dim;

    // NeoX RoPE: q_left_new  = q_left  * cos - q_right * sin
    //            q_right_new = q_right * cos + q_left  * sin
    for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
      float q_left = static_cast<float>(q_in[q_base + d]);
      float q_right = static_cast<float>(q_in[q_base + d + half_head_dim]);

      // cos_emb layout: [max_seq_len, rotary_dim], row stride = rotary_dim
      float cos_val =
          (d < rotary_dim) ? __ldg(&cos_emb[pos * rotary_dim + d]) : 1.0f;
      float sin_val =
          (d < rotary_dim) ? __ldg(&sin_emb[pos * rotary_dim + d]) : 0.0f;

      q_out[q_base + d] = static_cast<T>(q_left * cos_val - q_right * sin_val);
      q_out[q_base + d + half_head_dim] =
          static_cast<T>(q_right * cos_val + q_left * sin_val);
    }

  } else {
    // =====================================================================
    // K: Apply NeoX RoPE + write to key_cache + k_out
    // V: Write to value_cache (no RoPE)
    // =====================================================================
    const int kv_head_id = head_idx - num_heads;
    if (kv_head_id >= kv_num_heads) return;

    const int batch_id = batch_ids[token_id];
    const int block_idx_in_seq = static_cast<int>(pos / block_size);
    const int block_offset = static_cast<int>(pos % block_size);
    const int physical_block =
        __ldg(&block_tables[batch_id * max_blocks_per_seq + block_idx_in_seq]);

    if (physical_block < 0) return;  // Skip if block freed (preempted)

    const int64_t kv_base =
        static_cast<int64_t>(token_id) * kv_num_heads * head_dim +
        kv_head_id * head_dim;
    const int64_t cache_base = static_cast<int64_t>(physical_block) *
                                   kv_num_heads * block_size * head_dim +
                               kv_head_id * block_size * head_dim +
                               block_offset * head_dim;

    // K: NeoX RoPE + write to k_out and key_cache
    for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
      float k_left = static_cast<float>(k_in[kv_base + d]);
      float k_right = static_cast<float>(k_in[kv_base + d + half_head_dim]);

      float cos_val =
          (d < rotary_dim) ? __ldg(&cos_emb[pos * rotary_dim + d]) : 1.0f;
      float sin_val =
          (d < rotary_dim) ? __ldg(&sin_emb[pos * rotary_dim + d]) : 0.0f;

      T k_left_new = static_cast<T>(k_left * cos_val - k_right * sin_val);
      T k_right_new = static_cast<T>(k_right * cos_val + k_left * sin_val);

      k_out[kv_base + d] = k_left_new;
      k_out[kv_base + d + half_head_dim] = k_right_new;
      key_cache[cache_base + d] = k_left_new;
      key_cache[cache_base + d + half_head_dim] = k_right_new;
    }

    // V: vectorized copy to value_cache (no RoPE)
    // float4 = 16 bytes = 8 half values or 4 float values
    const int vec_size = 16 / sizeof(T);
    const int num_vecs = head_dim / vec_size;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
      const int offset = vi * vec_size;
      float4 v_val = *reinterpret_cast<const float4*>(&v_in[kv_base + offset]);
      *reinterpret_cast<float4*>(&value_cache[cache_base + offset]) = v_val;
    }

    // Handle remainder if head_dim is not divisible by vec_size
    const int rem_start = num_vecs * vec_size;
    for (int d = rem_start + threadIdx.x; d < head_dim; d += blockDim.x) {
      value_cache[cache_base + d] = v_in[kv_base + d];
    }
  }
}

// ============================================================================
// Paddle custom op host function
// ============================================================================

void V100RopeWriteCache(
    paddle::Tensor& q_out,               // pre-allocated, inplace output
    paddle::Tensor& k_out,               // pre-allocated, inplace output
    const paddle::Tensor& q,             // [num_tokens, num_heads, head_dim]
    const paddle::Tensor& k,             // [num_tokens, kv_num_heads, head_dim]
    const paddle::Tensor& v,             // [num_tokens, kv_num_heads, head_dim]
    const paddle::Tensor& cos_emb,       // [max_seq_len, rotary_dim]
    const paddle::Tensor& sin_emb,       // [max_seq_len, rotary_dim]
    paddle::Tensor& key_cache,           // inplace modified
    paddle::Tensor& value_cache,         // inplace modified
    const paddle::Tensor& block_tables,  // [batch_size, max_blocks_per_seq]
    const paddle::Tensor& positions,     // [num_tokens]
    const paddle::Tensor& batch_ids,     // [num_tokens]
    int num_heads,
    int kv_num_heads,
    int head_dim,
    int rotary_dim,
    int block_size,
    int max_blocks_per_seq) {
  auto stream = q.stream();
  const int num_tokens = q.dims()[0];
  const int THREADS = 128;

  // Grid: one block per (token, head).
  // Q heads and KV heads processed in separate blocks, no wasted work.
  dim3 grid(num_tokens, num_heads + kv_num_heads);
  dim3 block(THREADS);

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      q.dtype(), "v100_fused_rope_write_cache_kernel", [&] {
        v100_fused_rope_write_cache_kernel<data_t>
            <<<grid, block, 0, stream>>>(q.data<data_t>(),
                                         k.data<data_t>(),
                                         v.data<data_t>(),
                                         cos_emb.data<float>(),
                                         sin_emb.data<float>(),
                                         q_out.data<data_t>(),
                                         k_out.data<data_t>(),
                                         key_cache.data<data_t>(),
                                         value_cache.data<data_t>(),
                                         block_tables.data<int>(),
                                         positions.data<int64_t>(),
                                         batch_ids.data<int>(),
                                         num_tokens,
                                         num_heads,
                                         kv_num_heads,
                                         head_dim,
                                         rotary_dim,
                                         block_size,
                                         max_blocks_per_seq);
      });
}

// ============================================================================
// PD_BUILD_STATIC_OP registration (consistent with v100_decode_attention.cu)
// ============================================================================

PD_BUILD_STATIC_OP(v100_rope_write_cache)
    .Inputs({"q_out",
             "k_out",
             "q",
             "k",
             "v",
             "cos_emb",
             "sin_emb",
             "key_cache",
             "value_cache",
             "block_tables",
             "positions",
             "batch_ids"})
    .Outputs(
        {"q_out_result", "k_out_result", "key_cache_out", "value_cache_out"})
    .Attrs({"num_heads: int",
            "kv_num_heads: int",
            "head_dim: int",
            "rotary_dim: int",
            "block_size: int",
            "max_blocks_per_seq: int"})
    .SetInplaceMap({{"q_out", "q_out_result"},
                    {"k_out", "k_out_result"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(V100RopeWriteCache));
