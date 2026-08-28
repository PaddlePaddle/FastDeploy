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

#include "helper.h"
#include "paddle/extension.h"

template <typename T, int VecSize = 1>
__global__ void cache_kernel(const T* __restrict__ qkv,
                             T* __restrict__ key_cache,
                             T* __restrict__ value_cache,
                             const int* __restrict__ block_tables,
                             const int* __restrict__ batch_id_per_token,
                             const int* __restrict__ cu_seqlens_q,
                             const int* __restrict__ seq_lens_encoder,
                             const int* __restrict__ seq_lens_decoder,
                             const int max_seq_len,
                             const int max_blocks_per_seq,
                             const int num_heads,
                             const int head_size,
                             const int block_size,
                             const uint32_t elem_cnt,
                             const int kv_num_heads) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t hidden_size = kv_num_heads * head_size;
  const uint32_t offset = 2 * hidden_size;
  for (uint32_t linear_index = global_thread_idx * VecSize,
                step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / offset;
    const uint32_t bias = linear_index % offset;
    const uint32_t qkv_id = bias / hidden_size;  // skip q
    const uint32_t qkv_bias = bias % hidden_size;
    const uint32_t hi = qkv_bias / head_size;
    const uint32_t h_bias = qkv_bias % head_size;
    const int32_t ori_bi = batch_id_per_token[token_idx];
    if (ori_bi == -1) continue;  // skip batch_id_per_token[token_idx]=-1
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] == 0) continue;
    const uint32_t ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int32_t* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;

    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    const uint32_t tgt_idx = block_idx * block_size * kv_num_heads * head_size +
                             block_offset * kv_num_heads * head_size +
                             hi * head_size + h_bias;
    const uint32_t ori_idx =
        token_idx * (num_heads + 2 * kv_num_heads) * head_size +
        num_heads * head_size + qkv_id * hidden_size + hi * head_size + h_bias;
    Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (qkv_id == 0) {
      Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}

void WriteCacheKV(
    const paddle::Tensor&
        qkv,  // [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]
    const paddle::optional<paddle::Tensor>&
        seq_lens_encoder,                      // [max_batch_size, 1]
    const paddle::Tensor& seq_lens_decoder,    // [max_batch_size, 1]
    const paddle::Tensor& batch_id_per_token,  // [num_tokens,]
    const paddle::Tensor& cu_seqlens_q,        // [batch_size + 1,]
    const paddle::Tensor& block_tables,  // [max_batch_size, max_blocks_per_seq]
    paddle::Tensor&
        key_cache,  // [num_blocks, block_size, kv_num_heads, head_dim]
    paddle::Tensor&
        value_cache,  // [num_blocks, block_size, kv_num_heads, head_dim]
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int max_seq_len) {
  if (qkv.dtype() != paddle::DataType::BFLOAT16) {
    PD_THROW("Only support qkv dtype of BF16");
  }
  using data_t = phi::dtype::bfloat16;

  auto num_tokens = qkv.shape()[0];
  auto block_size = key_cache.shape()[1];
  auto max_blocks_per_seq = block_tables.shape()[1];

  const uint32_t elem_nums = num_tokens * 2 * kv_num_heads * head_dim;
  constexpr int PackSize = 16 / sizeof(data_t);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  launchWithPdlWhenEnabled(
      cache_kernel<data_t, PackSize>,
      grid_size,
      blocksize,
      0,
      qkv.stream(),
      qkv.data<data_t>(),
      key_cache.data<data_t>(),
      value_cache.data<data_t>(),
      block_tables.data<int>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      seq_lens_encoder ? seq_lens_encoder->data<int>() : nullptr,
      seq_lens_decoder.data<int>(),
      max_seq_len,
      max_blocks_per_seq,
      num_heads,
      head_dim,
      block_size,
      elem_nums,
      kv_num_heads);
}

PD_BUILD_STATIC_OP(write_cache_kv)
    .Inputs({"qkv",
             paddle::Optional("seq_lens_encoder"),
             "seq_lens_decoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "key_cache",
             "value_cache"})
    .Outputs({"key_cache_out", "value_cache_out"})
    .Attrs({"num_heads:int",
            "kv_num_heads:int",
            "head_dim:int",
            "max_seq_len:int"})
    .SetInplaceMap({{"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(WriteCacheKV));
