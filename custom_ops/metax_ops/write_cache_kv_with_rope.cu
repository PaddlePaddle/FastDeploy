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
__global__ void append_decode_cache_T_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT out_vec;
  LoadKVT cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int half_head_size = head_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int hi = bias / head_size;  // q + k + v
    const int h_bias = bias % head_size;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens_decoder[ori_bi];
    if (write_seq_id == 0) continue;
    const int* block_table_now = nullptr;
    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx =
        start_token_idx * hidden_size + hi * head_size + h_bias;
    const int bias_idx = hi * head_size + h_bias;
    Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_head_size + h_bias / 2;
      uint32_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size : emb_idx;
      Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }

#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);

      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        out_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        out_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        out_vec[2 * i] = src_vec[2 * i];
        out_vec[2 * i + 1] = src_vec[2 * i + 1];
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(out_vec, &qkv[ori_idx]);
    } else {
      // write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * head_size +
          block_offset * kv_num_heads * head_size + kv_head_idx * head_size +
          h_bias;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(out_vec, &key_cache[tgt_idx]);
      } else {
        Store<T, VecSize>(out_vec, &value_cache[tgt_idx]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_neox_partial_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int rotary_dim,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec, right_vec;
  LoadBiasT left_bias_vec, right_bias_vec;
  LoadKVT left_cache_vec, right_cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int half_rotary_dim = rotary_dim / 2;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int64_t half_hidden_size = hidden_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / half_hidden_size;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    if (hi < num_heads && h_bias >= half_rotary_dim) continue;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens_decoder[ori_bi];
    if (write_seq_id == 0) continue;
    const int* block_table_now = nullptr;
    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    uint32_t ori_idx_left =
        start_token_idx * hidden_size + hi * head_size + h_bias;
    uint32_t ori_idx_right = ori_idx_left + half_head_size;
    if (hi < num_heads) {
      ori_idx_right = ori_idx_left + half_rotary_dim;
    } else if (hi < num_heads + kv_num_heads) {
      if (h_bias < half_rotary_dim) {
        ori_idx_right = ori_idx_left + half_rotary_dim;
      } else {
        ori_idx_left = ori_idx_left + half_rotary_dim;
        ori_idx_right = ori_idx_left + half_rotary_dim;
      }
    }
    Load<T, VecSize>(&qkv[ori_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[ori_idx_right], &right_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_rotary_dim + h_bias;
      uint32_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size * 2 : emb_idx;
      if (h_bias < half_rotary_dim) {
        Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
        Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
      }
    }

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      if (hi < num_heads + kv_num_heads && h_bias < half_rotary_dim) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv[ori_idx_right]);
    } else {
      // write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      uint32_t tgt_idx_left =
          block_idx * kv_num_heads * block_size * head_size +
          block_offset * kv_num_heads * head_size + kv_head_idx * head_size +
          h_bias;
      uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        if (h_bias < half_rotary_dim) {
          tgt_idx_right = tgt_idx_left + half_rotary_dim;
        } else {
          tgt_idx_left = tgt_idx_left + half_rotary_dim;
          tgt_idx_right = tgt_idx_left + half_rotary_dim;
        }
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_neox_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec, right_vec;
  LoadBiasT left_bias_vec, right_bias_vec;
  LoadKVT left_cache_vec, right_cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int64_t half_hidden_size = hidden_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / half_hidden_size;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens_decoder[ori_bi];
    if (write_seq_id == 0) continue;
    const int* block_table_now = nullptr;
    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx_left =
        start_token_idx * hidden_size + hi * head_size + h_bias;
    const uint32_t ori_idx_right = ori_idx_left + half_head_size;
    Load<T, VecSize>(&qkv[ori_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[ori_idx_right], &right_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * head_size + h_bias;
      uint32_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size * 2 : emb_idx;
      Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv[ori_idx_right]);
    } else {
      // write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx_left =
          block_idx * kv_num_heads * block_size * head_size +
          block_offset * kv_num_heads * head_size + kv_head_idx * head_size +
          h_bias;
      const uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T>
void append_decode_cache_rope(T* qkv,
                              T* key_cache,
                              T* value_cache,
                              const int* block_tables,
                              const int* cu_seqlens_q,
                              const int* seq_lens_encoder,
                              const int* seq_lens_decoder,
                              const float* cos_emb,
                              const float* sin_emb,
                              const int max_seq_len,
                              const int max_blocks_per_seq,
                              const int num_heads,
                              const int kv_num_heads,
                              const int head_dim,
                              const int rotary_dim,
                              const int block_size,
                              const int batch_size,
                              const cudaStream_t& stream,
                              const bool use_neox_style,
                              const bool rope_3d) {
  const uint32_t elem_nums =
      use_neox_style
          ? batch_size * (num_heads + 2 * kv_num_heads) * head_dim / 2
          : batch_size * (num_heads + 2 * kv_num_heads) * head_dim;

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  if (use_neox_style) {
    if (rotary_dim < head_dim) {
      auto* kernelFn =
          append_decode_cache_T_neox_partial_rope_kernel<T, PackSize>;
      launchWithPdlWhenEnabled(kernelFn,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               qkv,
                               key_cache,
                               value_cache,
                               block_tables,
                               cu_seqlens_q,
                               seq_lens_encoder,
                               seq_lens_decoder,
                               cos_emb,
                               sin_emb,
                               max_seq_len,
                               max_blocks_per_seq,
                               num_heads,
                               head_dim,
                               rotary_dim,
                               block_size,
                               elem_nums,
                               kv_num_heads,
                               rope_3d);
    } else {
      auto* kernelFn = append_decode_cache_T_neox_rope_kernel<T, PackSize>;
      launchWithPdlWhenEnabled(kernelFn,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               qkv,
                               key_cache,
                               value_cache,
                               block_tables,
                               cu_seqlens_q,
                               seq_lens_encoder,
                               seq_lens_decoder,
                               cos_emb,
                               sin_emb,
                               max_seq_len,
                               max_blocks_per_seq,
                               num_heads,
                               head_dim,
                               block_size,
                               elem_nums,
                               kv_num_heads,
                               rope_3d);
    }
  } else {
    auto* kernelFn = append_decode_cache_T_rope_kernel<T, PackSize>;
    launchWithPdlWhenEnabled(kernelFn,
                             grid_size,
                             blocksize,
                             0,
                             stream,
                             qkv,
                             key_cache,
                             value_cache,
                             block_tables,
                             cu_seqlens_q,
                             seq_lens_encoder,
                             seq_lens_decoder,
                             cos_emb,
                             sin_emb,
                             max_seq_len,
                             max_blocks_per_seq,
                             num_heads,
                             head_dim,
                             block_size,
                             elem_nums,
                             kv_num_heads,
                             rope_3d);
  }
}

template <typename T, int VecSize = 1>
__global__ void append_speculate_cache_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int output_inner_dim,
    const int head_size,
    const int block_size,
    const int elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadFloat scale_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int half_head_size = head_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_id = linear_index / hidden_size;
    const int ori_bi = batch_id_per_token[token_id];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) continue;
    const int bias = linear_index % hidden_size;
    const int hi = bias / head_size;  // q + k + v
    const int h_bias = bias % head_size;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    const int write_seq_id =
        seq_lens_decoder[ori_bi] + token_id - start_token_idx;
    if (write_seq_id == 0) continue;
    const int* block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    if (block_idx < 0) continue;
    const int block_offset = write_seq_id % block_size;
    const int write_q_idx =
        token_id * output_inner_dim * head_size + hi * head_size + h_bias;
    const int bias_idx = hi * head_size + h_bias;
    Load<T, VecSize>(&qkv[linear_index], &src_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const int64_t emb_idx = write_seq_id * half_head_size + h_bias / 2;
      int64_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size : emb_idx;
      Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }

#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // add_bias + rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(bias_vec, &qkv[write_q_idx]);
    } else {
      //  write k/v
      const int kv_head_idx = (hi - num_heads) % kv_num_heads;
      const int tgt_idx = block_idx * kv_num_heads * block_size * head_size +
                          block_offset * kv_num_heads * head_size +
                          kv_head_idx * head_size + h_bias;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(bias_vec, &key_cache[tgt_idx]);
      } else {
        Store<T, VecSize>(bias_vec, &value_cache[tgt_idx]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_speculate_cache_neox_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int output_inner_dim,
    const int head_size,
    const int block_size,
    const int elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec, right_vec;
  LoadT left_bias_vec, right_bias_vec;
  LoadFloat left_out_scale_vec, right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int half_head_size = head_size / 2;
  const int64_t half_hidden_size = hidden_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_id = linear_index / half_hidden_size;
    const int ori_bi = batch_id_per_token[token_id];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) continue;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    const int write_seq_id =
        seq_lens_decoder[ori_bi] + token_id - start_token_idx;
    if (write_seq_id == 0) continue;
    const int* block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    if (block_idx < 0) continue;
    const int block_offset = write_seq_id % block_size;
    const int bias_idx_left = hi * head_size + h_bias;
    const int bias_idx_right = bias_idx_left + half_head_size;
    const int ori_idx_left = token_id * hidden_size + hi * head_size + h_bias;
    const int ori_idx_right = ori_idx_left + half_head_size;
    Load<T, VecSize>(&qkv[ori_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[ori_idx_right], &right_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const int64_t emb_idx = write_seq_id * head_size + h_bias;
      int64_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size * 2 : emb_idx;
      Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // add_bias + rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv[ori_idx_right]);
    } else {
      //  write k/v
      const int kv_head_idx = (hi - num_heads) % kv_num_heads;
      const int tgt_idx_left =
          block_idx * kv_num_heads * block_size * head_size +
          block_offset * kv_num_heads * head_size + kv_head_idx * head_size +
          h_bias;
      const uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_speculate_cache_neox_partial_rope_kernel(
    T* __restrict__ qkv,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int output_inner_dim,
    const int head_size,
    const int rotary_dim,
    const int block_size,
    const int elem_cnt,
    const int kv_num_heads,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec, right_vec;
  LoadT left_bias_vec, right_bias_vec;
  LoadFloat left_out_scale_vec, right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int half_head_size = head_size / 2;
  const int half_rotary_dim = rotary_dim / 2;
  const int64_t half_hidden_size = hidden_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_id = linear_index / half_hidden_size;
    const int ori_bi = batch_id_per_token[token_id];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] > 0) continue;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    if (hi < num_heads && h_bias >= half_rotary_dim) continue;
    const int start_token_idx = cu_seqlens_q[ori_bi];
    const int write_seq_id =
        seq_lens_decoder[ori_bi] + token_id - start_token_idx;
    if (write_seq_id == 0) continue;
    const int* block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    if (block_idx < 0) continue;
    const int block_offset = write_seq_id % block_size;
    const int bias_idx_left = hi * head_size + h_bias;
    const int bias_idx_right = bias_idx_left + half_head_size;
    int ori_idx_left = token_id * hidden_size + hi * head_size + h_bias;
    int ori_idx_right = ori_idx_left + half_head_size;
    if (hi < num_heads) {
      ori_idx_right = ori_idx_left + half_rotary_dim;
    } else if (hi < num_heads + kv_num_heads) {
      if (h_bias < half_rotary_dim) {
        ori_idx_right = ori_idx_left + half_rotary_dim;
      } else {
        ori_idx_left = ori_idx_left + half_rotary_dim;
        ori_idx_right = ori_idx_left + half_rotary_dim;
      }
    }
    Load<T, VecSize>(&qkv[ori_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[ori_idx_right], &right_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const int64_t emb_idx = write_seq_id * half_rotary_dim + h_bias;
      int64_t new_emb_idx =
          rope_3d ? emb_idx + ori_bi * max_seq_len * head_size * 2 : emb_idx;
      if (h_bias < half_rotary_dim) {
        Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
        Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
      }
    }

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      if (hi < num_heads + kv_num_heads && h_bias < half_rotary_dim) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }

    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv[ori_idx_right]);
    } else {
      //  write k/v
      const int kv_head_idx = (hi - num_heads) % kv_num_heads;
      int tgt_idx_left = block_idx * kv_num_heads * block_size * head_size +
                         block_offset * kv_num_heads * head_size +
                         kv_head_idx * head_size + h_bias;
      uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        if (h_bias < half_rotary_dim) {
          tgt_idx_right = tgt_idx_left + half_rotary_dim;
        } else {
          tgt_idx_left = tgt_idx_left + half_rotary_dim;
          tgt_idx_right = tgt_idx_left + half_rotary_dim;
        }
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T>
void append_speculate_cache_rope(T* qkv,
                                 T* key_cache,
                                 T* value_cache,
                                 const int* block_tables,
                                 const int* batch_id_per_token,
                                 const int* cu_seqlens_q,
                                 const int* seq_lens_encoder,
                                 const int* seq_lens_decoder,
                                 const float* cos_emb,
                                 const float* sin_emb,
                                 const int max_seq_len,
                                 const int max_blocks_per_seq,
                                 const int num_heads,
                                 const int kv_num_heads,
                                 const int dim_head,
                                 const int rotary_dim,
                                 const int block_size,
                                 const int token_num,
                                 const cudaStream_t& stream,
                                 const bool use_neox_style,
                                 const bool rope_3d) {
  int output_inner_dim = num_heads + 2 * kv_num_heads;

  const uint32_t elem_nums =
      use_neox_style ? token_num * (num_heads + 2 * kv_num_heads) * dim_head / 2
                     : token_num * (num_heads + 2 * kv_num_heads) * dim_head;
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int threads_per_block = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (use_neox_style) {
    if (rotary_dim < dim_head) {
      append_speculate_cache_neox_partial_rope_kernel<T, PackSize>
          <<<grid_size, threads_per_block, 0, stream>>>(qkv,
                                                        key_cache,
                                                        value_cache,
                                                        block_tables,
                                                        batch_id_per_token,
                                                        cu_seqlens_q,
                                                        seq_lens_encoder,
                                                        seq_lens_decoder,
                                                        cos_emb,
                                                        sin_emb,
                                                        max_seq_len,
                                                        max_blocks_per_seq,
                                                        num_heads,
                                                        output_inner_dim,
                                                        dim_head,
                                                        rotary_dim,
                                                        block_size,
                                                        elem_nums,
                                                        kv_num_heads,
                                                        rope_3d);
    } else {
      append_speculate_cache_neox_rope_kernel<T, PackSize>
          <<<grid_size, threads_per_block, 0, stream>>>(qkv,
                                                        key_cache,
                                                        value_cache,
                                                        block_tables,
                                                        batch_id_per_token,
                                                        cu_seqlens_q,
                                                        seq_lens_encoder,
                                                        seq_lens_decoder,
                                                        cos_emb,
                                                        sin_emb,
                                                        max_seq_len,
                                                        max_blocks_per_seq,
                                                        num_heads,
                                                        output_inner_dim,
                                                        dim_head,
                                                        block_size,
                                                        elem_nums,
                                                        kv_num_heads,
                                                        rope_3d);
    }
  } else {
    append_speculate_cache_rope_kernel<T, PackSize>
        <<<grid_size, threads_per_block, 0, stream>>>(qkv,
                                                      key_cache,
                                                      value_cache,
                                                      block_tables,
                                                      batch_id_per_token,
                                                      cu_seqlens_q,
                                                      seq_lens_encoder,
                                                      seq_lens_decoder,
                                                      cos_emb,
                                                      sin_emb,
                                                      max_seq_len,
                                                      max_blocks_per_seq,
                                                      num_heads,
                                                      output_inner_dim,
                                                      dim_head,
                                                      block_size,
                                                      elem_nums,
                                                      kv_num_heads,
                                                      rope_3d);
  }
}

void WriteCacheKVWithRoPE(
    paddle::Tensor&
        qkv,  // [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]
    const paddle::optional<paddle::Tensor>&
        seq_lens_encoder,                      // [max_batch_size, 1]
    const paddle::Tensor& seq_lens_decoder,    // [max_batch_size, 1]
    const paddle::Tensor& batch_id_per_token,  // [num_tokens,]
    const paddle::Tensor& cu_seqlens_q,        // [batch_size + 1,]
    const paddle::Tensor& block_tables,  // [max_batch_size, max_blocks_per_seq]
    const paddle::Tensor& rotary_embs,
    paddle::Tensor&
        key_cache,  // [num_blocks, block_size, kv_num_heads, head_dim]
    paddle::Tensor&
        value_cache,  // [num_blocks, block_size, kv_num_heads, head_dim]
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int max_seq_len,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const bool speculate_decoder) {
  if (qkv.dtype() != paddle::DataType::BFLOAT16) {
    PD_THROW("Only support qkv dtype of BF16");
  }
  using data_t = phi::dtype::bfloat16;

  const auto rotary_embs_shape = rotary_embs.shape();
  const auto num_tokens = qkv.shape()[0];
  const auto max_blocks_per_seq = block_tables.shape()[1];
  const auto batch_size = cu_seqlens_q.shape()[0] - 1;
  const auto block_size = key_cache.shape()[1];
  const auto rotary_dim = rotary_embs_shape[rotary_embs_shape.size() - 1] * 2;
  const float* cos_emb = rotary_embs.data<float>();
  const float* sin_emb =
      use_neox_rotary_style
          ? rotary_embs.data<float>() + max_seq_len * head_dim
          : rotary_embs.data<float>() + max_seq_len * head_dim / 2;
  if (rotary_dim < head_dim) {
    if (!use_neox_rotary_style) {
      PADDLE_THROW(
          phi::errors::Fatal("partial_rotary_factor < 1.0 only supports "
                             "neox_rotary_style=True"));
    }
    sin_emb = rotary_embs.data<float>() + max_seq_len * rotary_dim / 2;
  }

  if (!speculate_decoder) {
    append_decode_cache_rope(
        qkv.data<data_t>(),
        key_cache.data<data_t>(),
        value_cache.data<data_t>(),
        block_tables.data<int>(),
        cu_seqlens_q.data<int>(),
        seq_lens_encoder ? seq_lens_encoder->data<int>() : nullptr,
        seq_lens_decoder.data<int>(),
        cos_emb,
        sin_emb,
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads,
        head_dim,
        rotary_dim,
        block_size,
        batch_size,
        qkv.stream(),
        use_neox_rotary_style,
        rope_3d);
  } else {
    append_speculate_cache_rope(
        qkv.data<data_t>(),
        key_cache.data<data_t>(),
        value_cache.data<data_t>(),
        block_tables.data<int>(),
        batch_id_per_token.data<int>(),
        cu_seqlens_q.data<int>(),
        seq_lens_encoder ? seq_lens_encoder->data<int>() : nullptr,
        seq_lens_decoder.data<int>(),
        cos_emb,
        sin_emb,
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads,
        head_dim,
        rotary_dim,
        block_size,
        num_tokens,
        qkv.stream(),
        use_neox_rotary_style,
        rope_3d);
  }
}

PD_BUILD_STATIC_OP(write_cache_kv_with_rope)
    .Inputs({"qkv",
             paddle::Optional("seq_lens_encoder"),
             "seq_lens_decoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "rotary_embs",
             "key_cache",
             "value_cache"})
    .Outputs({"qkv_out", "key_cache_out", "value_cache_out"})
    .Attrs({"num_heads:int",
            "kv_num_heads:int",
            "head_dim:int",
            "max_seq_len:int",
            "use_neox_style:bool",
            "rope_3d:bool",
            "speculate_decoder:bool"})
    .SetInplaceMap({{"qkv", "qkv_out"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(WriteCacheKVWithRoPE));
