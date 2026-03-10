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
// Fuses two operations:
// 1. Apply RoPE to Q and K
// 2. Write K, V to paged block cache
//
// This replaces the Python implementations:
// - _python_apply_rope_to_qk()
// - _python_write_kv_to_block_cache()

#include "helper.h"
#include "paddle/extension.h"

// ============================================================================
// Fused RoPE + KV cache write kernel (NeoX style)
// ============================================================================
// Grid: (num_tokens), Block: (128 or 256)
// Each thread block handles one token's Q, K, V processing.

template <typename T, int VecSize = 4>
__global__ void V100FusedRopeWriteCacheKernel(
    const T* __restrict__ q_in,         // [num_tokens, num_heads, head_dim]
    const T* __restrict__ k_in,         // [num_tokens, kv_num_heads, head_dim]
    const T* __restrict__ v_in,         // [num_tokens, kv_num_heads, head_dim]
    const float* __restrict__ cos_emb,  // [max_seq_len, rotary_dim]
    const float* __restrict__ sin_emb,  // [max_seq_len, rotary_dim]
    T* __restrict__ q_out,              // [num_tokens, num_heads, head_dim]
    T* __restrict__ k_out,              // [num_tokens, kv_num_heads, head_dim]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_dim]
    T* __restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_dim]
    const int* __restrict__ block_tables,   // [batch_size, max_blocks_per_seq]
    const int64_t* __restrict__ positions,  // [num_tokens]
    const int* __restrict__ batch_ids,      // [num_tokens]
    const int num_tokens,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int rotary_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const bool use_neox_style) {
  const int token_id = blockIdx.x;
  if (token_id >= num_tokens) return;

  const int64_t pos = positions[token_id];
  const int batch_id = batch_ids[token_id];

  // Compute block cache destination
  const int block_idx = static_cast<int>(pos / block_size);
  const int block_offset = static_cast<int>(pos % block_size);
  const int physical_block =
      __ldg(&block_tables[batch_id * max_blocks_per_seq + block_idx]);

  // Load cos/sin for this position
  const int half_head_dim = head_dim / 2;
  const int half_rotary_dim = rotary_dim / 2;

  // Process Q heads
  for (int head_id = threadIdx.x; head_id < num_heads; head_id += blockDim.x) {
    // Source offset: q_in[token_id, head_id, :]
    const int64_t q_src_base =
        static_cast<int64_t>(token_id) * num_heads * head_dim +
        head_id * head_dim;

    // Apply NeoX RoPE to Q
    for (int d = 0; d < half_head_dim; d++) {
      float q_left = static_cast<float>(q_in[q_src_base + d]);
      float q_right = static_cast<float>(q_in[q_src_base + d + half_head_dim]);

      float cos_val, sin_val;
      if (d < half_rotary_dim) {
        cos_val = cos_emb[pos * half_rotary_dim + d];
        sin_val = sin_emb[pos * half_rotary_dim + d];
      } else {
        cos_val = 1.0f;
        sin_val = 0.0f;
      }

      float q_left_new, q_right_new;
      if (use_neox_style) {
        // NeoX style: [q1, q2] -> [q1*cos - q2*sin, q2*cos + q1*sin]
        q_left_new = q_left * cos_val - q_right * sin_val;
        q_right_new = q_right * cos_val + q_left * sin_val;
      } else {
        // Standard style: interleaved
        q_left_new = q_left * cos_val - q_right * sin_val;
        q_right_new = q_right * cos_val + q_left * sin_val;
      }

      q_out[q_src_base + d] = static_cast<T>(q_left_new);
      q_out[q_src_base + d + half_head_dim] = static_cast<T>(q_right_new);
    }
  }

  // Process KV heads
  for (int kv_head_id = threadIdx.x; kv_head_id < kv_num_heads;
       kv_head_id += blockDim.x) {
    // Source offset: k_in[token_id, kv_head_id, :]
    const int64_t k_src_base =
        static_cast<int64_t>(token_id) * kv_num_heads * head_dim +
        kv_head_id * head_dim;

    // Cache destination: cache[physical_block, kv_head_id, block_offset, :]
    const int64_t cache_dst_base = static_cast<int64_t>(physical_block) *
                                       kv_num_heads * block_size * head_dim +
                                   kv_head_id * block_size * head_dim +
                                   block_offset * head_dim;

    // Apply NeoX RoPE to K and write to cache
    for (int d = 0; d < half_head_dim; d++) {
      float k_left = static_cast<float>(k_in[k_src_base + d]);
      float k_right = static_cast<float>(k_in[k_src_base + d + half_head_dim]);

      float cos_val, sin_val;
      if (d < half_rotary_dim) {
        cos_val = cos_emb[pos * half_rotary_dim + d];
        sin_val = sin_emb[pos * half_rotary_dim + d];
      } else {
        cos_val = 1.0f;
        sin_val = 0.0f;
      }

      float k_left_new, k_right_new;
      if (use_neox_style) {
        k_left_new = k_left * cos_val - k_right * sin_val;
        k_right_new = k_right * cos_val + k_left * sin_val;
      } else {
        k_left_new = k_left * cos_val - k_right * sin_val;
        k_right_new = k_right * cos_val + k_left * sin_val;
      }

      // Write to k_out
      k_out[k_src_base + d] = static_cast<T>(k_left_new);
      k_out[k_src_base + d + half_head_dim] = static_cast<T>(k_right_new);

      // Write to key cache
      key_cache[cache_dst_base + d] = static_cast<T>(k_left_new);
      key_cache[cache_dst_base + d + half_head_dim] =
          static_cast<T>(k_right_new);
    }

    // Write V to cache (no RoPE)
    for (int d = 0; d < head_dim; d++) {
      T v_val = v_in[k_src_base + d];  // Same layout as K
      value_cache[cache_dst_base + d] = v_val;
    }
  }
}

// ============================================================================
// Optimized version with vectorized memory access
// ============================================================================

template <typename T>
__global__ void V100FusedRopeWriteCacheKernelVec4(
    const T* __restrict__ q_in,
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int64_t* __restrict__ positions,
    const int* __restrict__ batch_ids,
    const int num_tokens,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int rotary_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const bool use_neox_style) {
  // Grid: (num_tokens * max(num_heads, kv_num_heads))
  // Each thread block handles one (token, head) pair
  const int global_id = blockIdx.x;
  const int total_heads = max(num_heads, kv_num_heads);
  const int token_id = global_id / total_heads;
  const int head_id = global_id % total_heads;

  if (token_id >= num_tokens) return;

  const int64_t pos = positions[token_id];
  const int batch_id = batch_ids[token_id];

  const int block_idx = static_cast<int>(pos / block_size);
  const int block_offset = static_cast<int>(pos % block_size);
  const int physical_block =
      __ldg(&block_tables[batch_id * max_blocks_per_seq + block_idx]);

  const int half_head_dim = head_dim / 2;
  const int half_rotary_dim = rotary_dim / 2;

  // Process Q if head_id < num_heads
  if (head_id < num_heads) {
    const int64_t q_base =
        static_cast<int64_t>(token_id) * num_heads * head_dim +
        head_id * head_dim;

    for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
      float q_left = static_cast<float>(q_in[q_base + d]);
      float q_right = static_cast<float>(q_in[q_base + d + half_head_dim]);

      float cos_val =
          (d < half_rotary_dim) ? cos_emb[pos * half_rotary_dim + d] : 1.0f;
      float sin_val =
          (d < half_rotary_dim) ? sin_emb[pos * half_rotary_dim + d] : 0.0f;

      float q_left_new = q_left * cos_val - q_right * sin_val;
      float q_right_new = q_right * cos_val + q_left * sin_val;

      q_out[q_base + d] = static_cast<T>(q_left_new);
      q_out[q_base + d + half_head_dim] = static_cast<T>(q_right_new);
    }
  }

  // Process K, V if head_id < kv_num_heads
  if (head_id < kv_num_heads) {
    const int64_t kv_base =
        static_cast<int64_t>(token_id) * kv_num_heads * head_dim +
        head_id * head_dim;

    const int64_t cache_base = static_cast<int64_t>(physical_block) *
                                   kv_num_heads * block_size * head_dim +
                               head_id * block_size * head_dim +
                               block_offset * head_dim;

    // K with RoPE
    for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
      float k_left = static_cast<float>(k_in[kv_base + d]);
      float k_right = static_cast<float>(k_in[kv_base + d + half_head_dim]);

      float cos_val =
          (d < half_rotary_dim) ? cos_emb[pos * half_rotary_dim + d] : 1.0f;
      float sin_val =
          (d < half_rotary_dim) ? sin_emb[pos * half_rotary_dim + d] : 0.0f;

      float k_left_new = k_left * cos_val - k_right * sin_val;
      float k_right_new = k_right * cos_val + k_left * sin_val;

      k_out[kv_base + d] = static_cast<T>(k_left_new);
      k_out[kv_base + d + half_head_dim] = static_cast<T>(k_right_new);
      key_cache[cache_base + d] = static_cast<T>(k_left_new);
      key_cache[cache_base + d + half_head_dim] = static_cast<T>(k_right_new);
    }

    // V (no RoPE)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      T v_val = v_in[kv_base + d];
      value_cache[cache_base + d] = v_val;
    }
  }
}

// ============================================================================
// Paddle custom op interface
// ============================================================================

std::vector<paddle::Tensor> V100RopeWriteCache(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& cos_emb,
    const paddle::Tensor& sin_emb,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& positions,
    const paddle::Tensor& batch_ids,
    int num_heads,
    int kv_num_heads,
    int head_dim,
    int rotary_dim,
    int block_size,
    int max_blocks_per_seq,
    bool use_neox_style) {
  // Get dimensions
  const auto& q_dims = q.dims();
  const int num_tokens = q_dims[0];

  auto stream = q.stream();

  // Allocate output tensors
  paddle::Tensor q_out =
      GetEmptyTensor({num_tokens, num_heads, head_dim}, q.dtype(), q.place());
  paddle::Tensor k_out = GetEmptyTensor(
      {num_tokens, kv_num_heads, head_dim}, k.dtype(), k.place());

  // Get mutable cache pointers
  paddle::Tensor key_cache_mut = key_cache;
  paddle::Tensor value_cache_mut = value_cache;

  const int total_heads = std::max(num_heads, kv_num_heads);
  const int grid_size = num_tokens * total_heads;
  const int block_size_threads = 128;

  if (q.dtype() == paddle::DataType::FLOAT16) {
    typedef PDTraits<paddle::DataType::FLOAT16> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    V100FusedRopeWriteCacheKernelVec4<DataType_>
        <<<grid_size, block_size_threads, 0, stream>>>(
            reinterpret_cast<const DataType_*>(q.data<data_t>()),
            reinterpret_cast<const DataType_*>(k.data<data_t>()),
            reinterpret_cast<const DataType_*>(v.data<data_t>()),
            cos_emb.data<float>(),
            sin_emb.data<float>(),
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(key_cache_mut.data<data_t>()),
            reinterpret_cast<DataType_*>(value_cache_mut.data<data_t>()),
            block_tables.data<int>(),
            positions.data<int64_t>(),
            batch_ids.data<int>(),
            num_tokens,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_dim,
            block_size,
            max_blocks_per_seq,
            use_neox_style);
  } else if (q.dtype() == paddle::DataType::BFLOAT16) {
    // BF16 path (for compatibility, though V100 will convert to FP16)
    typedef PDTraits<paddle::DataType::BFLOAT16> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    V100FusedRopeWriteCacheKernelVec4<DataType_>
        <<<grid_size, block_size_threads, 0, stream>>>(
            reinterpret_cast<const DataType_*>(q.data<data_t>()),
            reinterpret_cast<const DataType_*>(k.data<data_t>()),
            reinterpret_cast<const DataType_*>(v.data<data_t>()),
            cos_emb.data<float>(),
            sin_emb.data<float>(),
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(key_cache_mut.data<data_t>()),
            reinterpret_cast<DataType_*>(value_cache_mut.data<data_t>()),
            block_tables.data<int>(),
            positions.data<int64_t>(),
            batch_ids.data<int>(),
            num_tokens,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_dim,
            block_size,
            max_blocks_per_seq,
            use_neox_style);
  } else {
    // FP32 fallback
    V100FusedRopeWriteCacheKernelVec4<float>
        <<<grid_size, block_size_threads, 0, stream>>>(
            q.data<float>(),
            k.data<float>(),
            v.data<float>(),
            cos_emb.data<float>(),
            sin_emb.data<float>(),
            q_out.data<float>(),
            k_out.data<float>(),
            key_cache_mut.data<float>(),
            value_cache_mut.data<float>(),
            block_tables.data<int>(),
            positions.data<int64_t>(),
            batch_ids.data<int>(),
            num_tokens,
            num_heads,
            kv_num_heads,
            head_dim,
            rotary_dim,
            block_size,
            max_blocks_per_seq,
            use_neox_style);
  }

  return {q_out, k_out};
}

PD_BUILD_OP(v100_rope_write_cache)
    .Inputs({"q",
             "k",
             "v",
             "cos_emb",
             "sin_emb",
             "key_cache",
             "value_cache",
             "block_tables",
             "positions",
             "batch_ids"})
    .Outputs({"q_out", "k_out"})
    .Attrs({"num_heads: int",
            "kv_num_heads: int",
            "head_dim: int",
            "rotary_dim: int",
            "block_size: int",
            "max_blocks_per_seq: int",
            "use_neox_style: bool"})
    .SetKernelFn(PD_KERNEL(V100RopeWriteCache));
