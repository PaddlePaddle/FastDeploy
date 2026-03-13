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

#include <cuda_runtime.h>

namespace fastdeploy {
namespace custom_ops {

/**
 * Launch Lightning Attention CUDA kernel
 * 
 * Lightning Attention 是 MiniMax-M1 模型的核心创新点，
 * 采用 block-wise 计算和指数衰减实现高效的线性注意力机制。
 * 
 * @param q Query tensor, shape: [batch, num_heads, seq_len, head_dim]
 * @param k Key tensor, shape: [batch, num_kv_heads, seq_len, head_dim]
 * @param v Value tensor, shape: [batch, num_kv_heads, seq_len, head_dim]
 * @param output Output tensor, shape: [batch, num_heads, seq_len, head_dim]
 * @param causal_mask Causal mask for attention (optional, can be nullptr)
 * @param scale Scaling factor, typically 1/sqrt(head_dim)
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of key/value heads (for GQA)
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param stream CUDA stream
 * @return cudaError_t CUDA error code
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
    cudaStream_t stream = 0);

/**
 * Lightning Attention backward kernel (for training)
 */
cudaError_t LaunchLightningAttentionBackward(
    const void* q,
    const void* k,
    const void* v,
    const void* grad_output,
    void* grad_q,
    void* grad_k,
    void* grad_v,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream = 0);

}  // namespace custom_ops
}  // namespace fastdeploy