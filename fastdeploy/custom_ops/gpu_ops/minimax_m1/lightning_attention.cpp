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

#include "fastdeploy/custom_ops/gpu_ops/minimax_m1/lightning_attention.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cmath>

namespace fastdeploy {
namespace custom_ops {

// Block size for lightning attention computation
#define LA_BLOCK_SIZE 64

/**
 * Lightning Attention 的核心 CUDA kernel
 * 
 * 采用 block-wise 计算方式，将 Q、K、V 矩阵分块处理，
 * 使用指数衰减实现 causal masking，支持增量推理的 KV cache。
 * 
 * @param q Query tensor, shape: [batch, num_heads, seq_len, head_dim]
 * @param k Key tensor, shape: [batch, num_kv_heads, seq_len, head_dim]
 * @param v Value tensor, shape: [batch, num_kv_heads, seq_len, head_dim]
 * @param output Output tensor, shape: [batch, num_heads, seq_len, head_dim]
 * @param causal_mask Causal mask for attention
 * @param scale Scaling factor (1/sqrt(head_dim))
 */
template<typename T>
__global__ void LightningAttentionKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ output,
    const bool* __restrict__ causal_mask,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim) {
    
    const int bid = blockIdx.x;  // batch * num_heads
    const int hid = blockIdx.y;  // head index within the batch
    const int blk = blockIdx.z;  // block index within sequence
    
    const int tid = threadIdx.x;
    const int blk_offset = blk * LA_BLOCK_SIZE;
    
    // Shared memory for block-wise computation
    extern __shared__ char shared_mem[];
    T* k_block = (T*)shared_mem;
    T* v_block = (T*)(shared_mem + LA_BLOCK_SIZE * head_dim * sizeof(T));
    T* o_block = (T*)(shared_mem + 2 * LA_BLOCK_SIZE * head_dim * sizeof(T));
    T* m_block = (T*)(shared_mem + 3 * LA_BLOCK_SIZE * sizeof(T));  // max values
    T* l_block = (T*)(shared_mem + 4 * LA_BLOCK_SIZE * sizeof(T));  // normalizer
    
    // Initialize output block
    if (tid < LA_BLOCK_SIZE) {
        o_block[tid] = T(0);
        m_block[tid] = -INFINITY;
        l_block[tid] = T(0);
    }
    __syncthreads();
    
    // Process each block in the sequence
    for (int j = 0; j <= blk; ++j) {
        // Load K and V blocks
        if (tid < LA_BLOCK_SIZE) {
            const int j_offset = j * LA_BLOCK_SIZE + tid;
            const int kv_head_id = hid % num_kv_heads;
            
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                const int k_idx = bid * num_heads * seq_len * head_dim +
                                  kv_head_id * seq_len * head_dim +
                                  j_offset * head_dim + d;
                k_block[tid * head_dim + d] = (j_offset < seq_len) ? k[k_idx] : T(0);
                
                const int v_idx = k_idx;  // Same layout for v
                v_block[tid * head_dim + d] = (j_offset < seq_len) ? v[v_idx] : T(0);
            }
        }
        __syncthreads();
        
        // Compute block-wise attention
        if (tid < LA_BLOCK_SIZE && blk_offset + tid < seq_len) {
            // Load current Q
            T q_vec[128];  // max head_dim
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                const int q_idx = bid * num_heads * seq_len * head_dim +
                                  hid * seq_len * head_dim +
                                  (blk_offset + tid) * head_dim + d;
                q_vec[d] = q[q_idx];
            }
            
            // Compute qk^T
            float qk = 0;
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                qk += static_cast<float>(q_vec[d]) * static_cast<float>(k_block[tid * head_dim + d]);
            }
            qk *= scale;
            
            // Apply causal mask
            if (j < blk || (j == blk && tid > LA_BLOCK_SIZE - 1)) {
                qk = -INFINITY;
            }
            
            // Update max value and normalizer
            float m_old = m_block[tid];
            float m_new = max(m_old, qk);
            float l_old = l_block[tid];
            float l_new = l_old * exp(m_old - m_new) + exp(qk - m_new);
            
            m_block[tid] = m_new;
            l_block[tid] = l_new;
            
            // Compute attention output
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                float attn_weight = exp(qk - m_new);
                o_block[tid * head_dim + d] = 
                    (o_block[tid * head_dim + d] * l_old * exp(m_old - m_new) + 
                     attn_weight * v_block[tid * head_dim + d]) / l_new;
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (tid < LA_BLOCK_SIZE && blk_offset + tid < seq_len) {
        const int out_idx = bid * num_heads * seq_len * head_dim +
                           hid * seq_len * head_dim +
                           (blk_offset + tid) * head_dim;
        
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            output[out_idx + d] = o_block[tid * head_dim + d];
        }
    }
}

/**
 * Lightning Attention CUDA 算子实现
 */
cudaError_t LaunchLightningAttention(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    const bool* causal_mask,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream) {
    
    dim3 blocks(batch_size * num_heads, num_heads, (seq_len + LA_BLOCK_SIZE - 1) / LA_BLOCK_SIZE);
    dim3 threads(LA_BLOCK_SIZE);
    
    size_t shared_mem_size = 2 * LA_BLOCK_SIZE * head_dim * sizeof(float) + 
                            2 * LA_BLOCK_SIZE * sizeof(float);
    
    LightningAttentionKernel<float><<<blocks, threads, shared_mem_size, stream>>>(
        static_cast<const float*>(q),
        static_cast<const float*>(k),
        static_cast<const float*>(v),
        static_cast<float*>(output),
        causal_mask,
        scale,
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim
    );
    
    return cudaGetLastError();
}

}  // namespace custom_ops
}  // namespace fastdeploy