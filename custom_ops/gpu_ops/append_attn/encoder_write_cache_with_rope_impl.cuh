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

#include "helper.h"
#include "mem_util.cuh"
#include "mma_tensor_op.cuh"
#include "utils.cuh"

template <typename T, int VecSize = 1>
__global__ void IntVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int new_emb_idx = rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    Load<int, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (qkv_id < 2) {
      Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                    static_cast<float>(bias_vec[2 * i])
                              : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                     static_cast<float>(bias_vec[2 * i + 1])
                               : input_right * out_scale_vec[2 * i + 1];
      if (qkv_id < 2) {  // qk rope
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
    Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int new_emb_idx = rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int64_t base_idx = token_idx * 3 * hidden_size +
                             qkv_id * hidden_size + hi * last_dim + h_bias;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void IntNeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    int new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len * 2 : emb_idx;
    const int bias_idx_left =
        qkv_id * full_hidden_size + hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx_left], &left_out_scale_vec);
    Load<float, VecSize>(&qkv_out_scales[bias_idx_right], &right_out_scale_vec);
    if (qkv_id < 2) {
      Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                                    static_cast<float>(left_bias_vec[i])
                              : input_left * left_out_scale_vec[i];
      input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                                     static_cast<float>(right_bias_vec[i])
                               : input_right * right_out_scale_vec[i];
      if (qkv_id < 2) {  // qk rope
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
    Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    int new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len * 2 : emb_idx;
    const int base_idx_left = token_idx * 3 * full_hidden_size +
                              qkv_id * full_hidden_size + hi * last_dim +
                              h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void IntGQAVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    Load<int, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (hi < q_num_head + kv_num_head) {
      Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                    static_cast<float>(bias_vec[2 * i])
                              : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                     static_cast<float>(bias_vec[2 * i + 1])
                               : input_right * out_scale_vec[2 * i + 1];
      if (hi < q_num_head + kv_num_head) {  // qk rope
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
    Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryQKNormKernel(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d,
    const float *q_norm_weight,
    const float *k_norm_weight,
    const float rms_norm_eps) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  LoadFloat tmp_vec;
  LoadFloat q_norm_vec, k_norm_vec;
  int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
  int64_t all_warp_num = gridDim.x * blockDim.y;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * last_dim;
  const int all_head_num = elem_cnt / last_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int global_hi = global_warp_idx; global_hi < all_head_num;
       global_hi += all_warp_num) {
    int64_t linear_index = global_hi * last_dim + threadIdx.x * VecSize;
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];
    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);

    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);

    float thread_m2 = 0.0f;
    float warp_m2 = 0.0f;

#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      float tmp1 = input_left * cos_tmp - input_right * sin_tmp;
      float tmp2 = input_right * cos_tmp + input_left * sin_tmp;
      tmp_vec[2 * i] = tmp1;
      tmp_vec[2 * i + 1] = tmp2;
      thread_m2 += tmp1 * tmp1 + tmp2 * tmp2;
    }
    WelfordWarpAllReduce<float, 32>(thread_m2, &warp_m2);
    float row_variance = max(warp_m2 / last_dim, 0.0f);
    float row_inv_var = Rsqrt(row_variance + rms_norm_eps);

    if (hi < q_num_head) {
      Load<float, VecSize>(&q_norm_weight[threadIdx.x * VecSize], &q_norm_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        src_vec[i] = static_cast<T>(tmp_vec[i] * row_inv_var * q_norm_vec[i]);
      }
    } else {
      Load<float, VecSize>(&k_norm_weight[threadIdx.x * VecSize], &k_norm_vec);
      for (int i = 0; i < VecSize; i++) {
        src_vec[i] = static_cast<T>(tmp_vec[i] * row_inv_var * k_norm_vec[i]);
      }
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(const T *qkv,
                                              const float *cos_emb,
                                              const float *sin_emb,
                                              const int *batch_id_per_token,
                                              const int *cu_seqlens_q,
                                              const int *seq_lens,
                                              const int *seq_lens_decoder,
                                              T *qkv_out,
                                              const int64_t elem_cnt,
                                              const int q_num_head,
                                              const int kv_num_head,
                                              const int seq_len,
                                              const int last_dim,
                                              const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * last_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    ;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);

    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void IntGQAVariableLengthRotaryQuantKVKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const float *qkv_out_scales,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const T *qkv_biases,
    const T *cache_k_scales,
    const T *cache_v_scales,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadIn = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  LoadIn src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  // const int hidden_size = num_head * last_dim;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    Load<int, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                    static_cast<float>(bias_vec[2 * i])
                              : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                     static_cast<float>(bias_vec[2 * i + 1])
                               : input_right * out_scale_vec[2 * i + 1];
      if (hi < q_num_head) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else if (hi < q_num_head + kv_num_head) {
        int k_hi = hi - q_num_head;
        const int scale_idx = k_hi * last_dim + h_bias;
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>((input_left * cos_tmp - input_right * sin_tmp) *
                           float(cache_k_scales[scale_idx + 2 * i]));
        bias_vec[2 * i + 1] =
            static_cast<T>((input_right * cos_tmp + input_left * sin_tmp) *
                           float(cache_k_scales[scale_idx + 2 * i + 1]));
      } else {
        int v_hi = hi - q_num_head - kv_num_head;
        const int scale_idx = v_hi * last_dim + h_bias;
        bias_vec[2 * i] = static_cast<T>(
            input_left * float(cache_v_scales[scale_idx + 2 * i]));
        bias_vec[2 * i + 1] = static_cast<T>(
            input_right * float(cache_v_scales[scale_idx + 2 * i + 1]));
      }
    }
    Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryQuantKVKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const T *qkv_biases,
    const T *cache_k_scales,
    const T *cache_v_scales,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  // const int hidden_size = num_head * last_dim;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          qkv_biases ? static_cast<float>(src_vec[2 * i] + bias_vec[2 * i])
                     : static_cast<float>(src_vec[2 * i]);
      const float input_right =
          qkv_biases
              ? static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1])
              : static_cast<float>(src_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);
      if (hi < q_num_head) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else if (hi < q_num_head + kv_num_head) {
        int k_hi = hi - q_num_head;
        const int scale_idx = k_hi * last_dim + h_bias;
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec[2 * i] =
            static_cast<T>((input_left * cos_tmp - input_right * sin_tmp) *
                           float(cache_k_scales[scale_idx + 2 * i]));
        src_vec[2 * i + 1] =
            static_cast<T>((input_right * cos_tmp + input_left * sin_tmp) *
                           float(cache_k_scales[scale_idx + 2 * i + 1]));
      } else {
        int v_hi = hi - q_num_head - kv_num_head;
        const int scale_idx = v_hi * last_dim + h_bias;
        src_vec[2 * i] = static_cast<T>(
            input_left * float(cache_v_scales[scale_idx + 2 * i]));
        src_vec[2 * i + 1] = static_cast<T>(
            input_right * float(cache_v_scales[scale_idx + 2 * i + 1]));
      }
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void IntGQANeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * half_lastdim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    int new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len * 2 : emb_idx;
    const int bias_idx_left = hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx_left], &left_out_scale_vec);
    Load<float, VecSize>(&qkv_out_scales[bias_idx_right], &right_out_scale_vec);
    if (hi < (q_num_head + kv_num_head)) {
      Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                                    static_cast<float>(left_bias_vec[i])
                              : input_left * left_out_scale_vec[i];
      input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                                     static_cast<float>(right_bias_vec[i])
                               : input_right * right_out_scale_vec[i];
      if (hi < (q_num_head + kv_num_head)) {  // qk rope
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
    Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(const T *qkv,
                                                  const float *cos_emb,
                                                  const float *sin_emb,
                                                  const int *batch_id_per_token,
                                                  const int *cu_seqlens_q,
                                                  const int *seq_lens,
                                                  const int *seq_lens_decoder,
                                                  const float *qkv_out_scales,
                                                  const T *qkv_biases,
                                                  T *qkv_out,
                                                  const int64_t elem_cnt,
                                                  const int q_num_head,
                                                  const int kv_num_head,
                                                  const int seq_len,
                                                  const int last_dim,
                                                  const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * half_lastdim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len * 2 : emb_idx;
    const int base_idx_left =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthPartialRotaryKernel(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int head_dim,
    const int rotary_dim,
    const bool rope_3d) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int rotary_dim_half = rotary_dim / 2;
  const int offset = (q_num_head + kv_num_head) * rotary_dim_half;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / rotary_dim_half;
    const int h_bias = bias % rotary_dim_half;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * rotary_dim_half + h_bias;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * head_dim * seq_len * 2 : emb_idx;
    const int base_idx_left =
        token_idx * (q_num_head + 2 * kv_num_head) * head_dim + hi * head_dim +
        h_bias;
    const int base_idx_right = base_idx_left + rotary_dim_half;

    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
    Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, int VecSize = 1>
__global__ void cache_kernel(
    const T *__restrict__ qkv,    // [num_tokens, num_heads + 2 * kv_num_heads,
                                  // head_size]
    T *__restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size]
    T *__restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size]
    const int *__restrict__ block_tables,        // [bsz, max_blocks_per_seq]
    const int *__restrict__ batch_id_per_token,  // [num_tokens]
    const int *__restrict__ cu_seqlens_q,        // [bsz]
    const int *__restrict__ seq_lens,            // [bsz]
    const int *__restrict__ seq_lens_decoder,    // [bsz]
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
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
    if (seq_lens[ori_bi] == 0) continue;
    const uint32_t ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) + seq_lens_decoder[ori_bi];

    const int32_t *block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;

    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * head_size +
                             hi * block_size * head_size +
                             block_offset * head_size + h_bias;
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS,
          bool is_need_kv_quant,
          bool IsFP8 = false>
__global__ void append_write_cache_kv_c8_qkv(
    uint8_t *__restrict__ cache_k,
    uint8_t *__restrict__ cache_v,
    const T *__restrict__ qkv_input,
    const T *__restrict__ cache_k_scales,
    const T *__restrict__ cache_v_scales,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ batch_id_per_token,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ block_tables,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int kv_num_heads) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t pad_len = BLOCK_SIZE;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const T cache_k_scale = cache_k_scales[kv_head_idx];
  const T cache_v_scale = cache_v_scales[kv_head_idx];
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids[btid];
  const uint32_t seq_len_this_time = seq_lens_this_time[batch_id];
  if (seq_len_this_time <= 0) {
    return;
  }
  const int *block_table_now = nullptr;

  block_table_now = block_tables + batch_id * max_blocks_per_seq;

  const uint32_t num_rows_per_block =
      NUM_WARPS * num_frags_z * 16;  // BLOCK_SIZE
  const uint32_t start_len = seq_lens_decoder[batch_id];
  const uint32_t bf_pad_len = start_len % pad_len;
  const uint32_t start_len_pad = start_len - bf_pad_len;
  const uint32_t end_len = start_len + seq_len_this_time;

  const uint32_t tile_start = start_len_pad + tile_id * num_rows_per_block;
  int block_id = __ldg(&block_table_now[tile_start / BLOCK_SIZE]);
  uint32_t chunk_start = tile_start + wid * num_frags_z * 16 + tid / 8;

  const uint32_t start_token_idx = cu_seqlens_q[batch_id];
  const uint32_t kv_batch_stride = (num_heads + 2 * kv_num_heads) * HEAD_DIM;
  const uint32_t kv_h_stride = HEAD_DIM;
  __shared__ T k_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_smem_ori[num_rows_per_block * HEAD_DIM];
  if (tile_start >= start_len) {
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    // int lane_id = wid * 32 + tid;
    // pad zero for this kv_head_idx for this block
    LoadPadKVT pad_cache_vec;
    *(reinterpret_cast<uint4 *>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
    // reset k
    constexpr int num_vecs_per_head_k = HEAD_DIM / KV_VEC_SIZE;
    constexpr int num_token_each_time_k = 32 / num_vecs_per_head_k;
    uint32_t tgt_idx =
        (block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE * HEAD_DIM +
        tid % num_vecs_per_head_k * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_k; block_i < BLOCK_SIZE;
         block_i += num_token_each_time_k) {
      Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                  &cache_k[tgt_idx + block_i * HEAD_DIM]);
    }

    // reset v
    const int num_vecs_per_head_v = BLOCK_SIZE / KV_VEC_SIZE;
    const int num_token_each_time_v = 32 / num_vecs_per_head_v;
    tgt_idx = (block_id * kv_num_heads + kv_head_idx) * HEAD_DIM * BLOCK_SIZE +
              tid % num_vecs_per_head_v * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_v; block_i < HEAD_DIM;
         block_i += num_token_each_time_v) {
      Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                  &cache_v[tgt_idx + block_i * BLOCK_SIZE]);
    }
  }
  smem_t k_smem(k_smem_ori);
  smem_t v_smem(v_smem_ori);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid / 8, tid % 8);  // 4 * 8 per warp

  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  constexpr uint32_t num_frags_v = num_frags_y / NUM_WARPS;
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, wid * num_frags_v * 2 + tid / 16);

  // load kv gmem to smem
  const uint32_t real_start_token_idx = start_token_idx - bf_pad_len +
                                        tile_id * num_rows_per_block +
                                        wid * num_frags_z * 16 + tid / 8;
  uint32_t k_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
  uint32_t v_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y / 4;
           ++fy) {  // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        if (chunk_start >= start_len && chunk_start < end_len) {
          k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + k_read_idx, chunk_start < end_len);
          v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + v_read_idx, chunk_start < end_len);
        }
        kv_smem_offset_w =
            k_smem.advance_offset_by_column<8>(kv_smem_offset_w, fy);
        k_read_idx += 8 * num_elems_per_128b<T>();
        v_read_idx += 8 * num_elems_per_128b<T>();
      }
      kv_smem_offset_w =
          k_smem.advance_offset_by_row<4, num_vecs_per_head>(kv_smem_offset_w) -
          2 * num_frags_y;
      chunk_start += 4;
      k_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      v_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
    }
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // mask, quant, store
  using LoadKVT = AlignedVector<uint8_t, 4>;
  LoadKVT cache_vec1;
  LoadKVT cache_vec2;

  uint32_t chunk_start_k = tile_start + wid * num_frags_z * 16 + tid / 4;
  uint32_t kv_frag[4];
  const uint32_t write_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_b_stride = HEAD_DIM;
  const uint32_t write_d_stride = BLOCK_SIZE;
  uint32_t k_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_z * 16 + tid / 4) * write_b_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
    uint32_t k_write_idx_now_z = k_write_idx + fz * 16 * write_b_stride;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t k_write_idx_now = k_write_idx_now_z +
                                 fy % 2 * 8 * write_b_stride +
                                 fy / 2 * 32;  // + fy % 2 * 16;
      // load
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // quant
      T *k_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_k + k_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_k + k_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_k + (v_id / 4) * 8 >= start_len &&
            chunk_start_k + (v_id / 4) * 8 < end_len) {
          uint_quant_value = QuantToC8<T, is_need_kv_quant, IsFP8>(
              cache_k_scale, k_frag_T[v_id], 127.0f, -127.0f);
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id - 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_k + k_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_k + k_write_idx_now + 16);
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16, num_vecs_per_head>(k_smem_offset_r) -
        2 * num_frags_y;
    chunk_start_k += 16;
  }

  uint32_t chunk_start_v = tile_start + tid % 4 * 2;
  uint32_t v_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_v * 16 + tid / 4) * write_d_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
  const uint32_t num_frags_z_v = num_frags_z * NUM_WARPS;
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_v; ++fy) {
    uint32_t v_write_idx_now_v = v_write_idx + fy * 16 * write_d_stride;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z_v; ++fz) {
      uint32_t v_write_idx_now = v_write_idx_now_v +
                                 fz % 2 * 8 * write_d_stride +
                                 fz / 2 * 32;  // + fz % 2 * 16;
      // load
      v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, kv_frag);
      // quant
      T *v_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_v + v_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_v + v_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 >= start_len &&
            chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 < end_len) {
          uint_quant_value = QuantToC8<T, is_need_kv_quant, IsFP8>(
              cache_v_scale, v_frag_T[v_id], 127.0f, -127.0f);
          // store now
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id % 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_v + v_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_v + v_write_idx_now + 16);
      chunk_start_v += 16;
      v_smem_offset_r =
          k_smem.advance_offset_by_row<16, num_vecs_per_head>(v_smem_offset_r);
    }
    v_smem_offset_r = k_smem.advance_offset_by_column<2>(
                          v_smem_offset_r, wid * num_frags_v + fy) -
                      16 * num_frags_z_v * num_vecs_per_head;
    chunk_start_v -= 16 * num_frags_z_v;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS,
          bool is_need_kv_quant,
          bool IsFP8 = true>
__global__ void append_write_cache_kv_c8_qkv_dynamic(
    uint8_t *__restrict__ cache_k,
    uint8_t *__restrict__ cache_v,
    const T *__restrict__ qkv_input,
    T *__restrict__ cache_k_scales,  // [block_num, num_heads, block_size]
    T *__restrict__ cache_v_scales,  // [block_num, num_heads, block_size]
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ batch_id_per_token,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ block_tables,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int kv_num_heads) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t pad_len = BLOCK_SIZE;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const T cache_k_scale = cache_k_scales[kv_head_idx];
  const T cache_v_scale = cache_v_scales[kv_head_idx];
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids[btid];
  const uint32_t seq_len_this_time = seq_lens_this_time[batch_id];
  if (seq_len_this_time <= 0) {
    return;
  }
  const int *block_table_now = nullptr;

  block_table_now = block_tables + batch_id * max_blocks_per_seq;

  const uint32_t num_rows_per_block =
      NUM_WARPS * num_frags_z * 16;  // BLOCK_SIZE
  const uint32_t start_len = seq_lens_decoder[batch_id];
  const uint32_t bf_pad_len = start_len % pad_len;
  const uint32_t start_len_pad = start_len - bf_pad_len;
  const uint32_t end_len = start_len + seq_len_this_time;

  const uint32_t tile_start = start_len_pad + tile_id * num_rows_per_block;
  int block_id = __ldg(&block_table_now[tile_start / BLOCK_SIZE]);
  uint32_t chunk_start = tile_start + wid * num_frags_z * 16 + tid / 8;

  const uint32_t start_token_idx = cu_seqlens_q[batch_id];
  const uint32_t kv_batch_stride = (num_heads + 2 * kv_num_heads) * HEAD_DIM;
  const uint32_t kv_h_stride = HEAD_DIM;
  __shared__ T k_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_scale_smem[BLOCK_SIZE];
  if (tile_start >= start_len) {
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    // pad zero for this kv_head_idx for this block
    LoadPadKVT pad_cache_vec;
    *(reinterpret_cast<uint4 *>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
    // reset k
    constexpr int num_vecs_per_head_k = HEAD_DIM / KV_VEC_SIZE;
    constexpr int num_token_each_time_k = 32 / num_vecs_per_head_k;
    uint32_t tgt_idx =
        (block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE * HEAD_DIM +
        tid % num_vecs_per_head_k * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_k; block_i < BLOCK_SIZE;
         block_i += num_token_each_time_k) {
      Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                  &cache_k[tgt_idx + block_i * HEAD_DIM]);
    }

    // reset v
    const int num_vecs_per_head_v = BLOCK_SIZE / KV_VEC_SIZE;
    const int num_token_each_time_v = 32 / num_vecs_per_head_v;
    tgt_idx = (block_id * kv_num_heads + kv_head_idx) * HEAD_DIM * BLOCK_SIZE +
              tid % num_vecs_per_head_v * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_v; block_i < HEAD_DIM;
         block_i += num_token_each_time_v) {
      Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                  &cache_v[tgt_idx + block_i * BLOCK_SIZE]);
    }
  }
  smem_t k_smem(k_smem_ori);
  smem_t v_smem(v_smem_ori);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid / 8, tid % 8);  // 4 * 8 per warp

  /*
   0 | 1
   2 | 3
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  constexpr uint32_t num_frags_v = num_frags_y / NUM_WARPS;
  /*
   0 | 2
   1 | 3
  */
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, wid * num_frags_v * 2 + tid / 16);

  // load kv gmem to smem
  const uint32_t real_start_token_idx = start_token_idx - bf_pad_len +
                                        tile_id * num_rows_per_block +
                                        wid * num_frags_z * 16 + tid / 8;
  uint32_t k_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
  uint32_t v_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y / 4;
           ++fy) {  // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        if (chunk_start >= start_len && chunk_start < end_len) {
          k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + k_read_idx, chunk_start < end_len);
          v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + v_read_idx, chunk_start < end_len);
        }
        kv_smem_offset_w =
            k_smem.advance_offset_by_column<8>(kv_smem_offset_w, fy);
        k_read_idx += 8 * num_elems_per_128b<T>();
        v_read_idx += 8 * num_elems_per_128b<T>();
      }
      kv_smem_offset_w =
          k_smem.advance_offset_by_row<4, num_vecs_per_head>(kv_smem_offset_w) -
          2 * num_frags_y;
      chunk_start += 4;
      k_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      v_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
    }
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // reduce scale
  // 16 rows per warp
  uint32_t kv_reduce_frag[4];
  T *kv_reduce_frag_T = reinterpret_cast<T *>(kv_reduce_frag);

  T k_local_max_value[num_frags_z * 2];
  T v_local_max_value[num_frags_z * 2];
#pragma unroll
  for (int i = 0; i < num_frags_z * 2; i++) {
    k_local_max_value[i] = -INFINITY;
  }
#pragma unroll
  for (int i = 0; i < num_frags_z * 2; i++) {
    v_local_max_value[i] = -INFINITY;
  }
  const int num_kv_heads = gridDim.z;
  const int scale_offset =
      block_id * num_kv_heads * BLOCK_SIZE + kv_head_idx * BLOCK_SIZE;
  T *cache_k_scale_now = cache_k_scales + scale_offset;
  T *cache_v_scale_now = cache_v_scales + scale_offset;
  // k scale
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // reduce per thread, 4 threads each row
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_reduce_frag);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        k_local_max_value[fz * 2] =
            __hmax(__habs(kv_reduce_frag_T[i]), k_local_max_value[fz * 2]);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        k_local_max_value[fz * 2 + 1] = __hmax(__habs(kv_reduce_frag_T[i + 4]),
                                               k_local_max_value[fz * 2 + 1]);
      }
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    // reduce per row
    for (int i = 0; i < 2; i++) {
      T local_max_value = __habs(k_local_max_value[fz * 2 + i]);
      local_max_value = __hmax(local_max_value,
                               __shfl_xor_sync(0xffffffff, local_max_value, 2));
      local_max_value = __hmax(local_max_value,
                               __shfl_xor_sync(0xffffffff, local_max_value, 1));
      // used for quant
      k_local_max_value[fz * 2 + i] = __hdiv(448, local_max_value);
    }
    // store
    if (tid % 4 == 0) {
      const int offset_now = wid * num_frags_z * 16 + tid / 4;
      // used for dequant
      if (tile_start + offset_now >= start_len) {
        if (tile_start + offset_now < end_len) {
          cache_k_scale_now[offset_now] = __hdiv(1, k_local_max_value[fz * 2]);
        } else {
          cache_k_scale_now[offset_now] = 0;
        }
      }
      if (tile_start + offset_now + 8 >= start_len) {
        if (tile_start + offset_now + 8 < end_len) {
          cache_k_scale_now[offset_now + 8] =
              __hdiv(1, k_local_max_value[fz * 2 + 1]);
        } else {
          cache_k_scale_now[offset_now + 8] = 0;
        }
      }
    }
    __syncthreads();
    k_smem_offset_r -= 2 * num_frags_y;  // num_frags_z = 1
  }
// v scale
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // reduce per thread, 4 threads each row
      v_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_reduce_frag);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v_local_max_value[fz * 2] =
            __hmax(__habs(kv_reduce_frag_T[i]), v_local_max_value[fz * 2]);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v_local_max_value[fz * 2 + 1] = __hmax(__habs(kv_reduce_frag_T[i + 4]),
                                               v_local_max_value[fz * 2 + 1]);
      }
      k_smem_offset_r = v_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    // reduce per row
    for (int i = 0; i < 2; i++) {
      T local_max_value = __habs(v_local_max_value[fz * 2 + i]);
      local_max_value = __hmax(local_max_value,
                               __shfl_xor_sync(0xffffffff, local_max_value, 2));
      local_max_value = __hmax(local_max_value,
                               __shfl_xor_sync(0xffffffff, local_max_value, 1));
      v_local_max_value[fz * 2 + i] = __hdiv(448, local_max_value);
    }
    // store
    if (tid % 4 == 0) {
      const int offset_now = wid * num_frags_z * 16 + tid / 4;
      // used for dequant
      if (tile_start + offset_now >= start_len) {
        if (tile_start + offset_now < end_len) {
          cache_v_scale_now[offset_now] = __hdiv(1, v_local_max_value[fz * 2]);
          v_scale_smem[offset_now] = v_local_max_value[fz * 2];
        } else {
          cache_v_scale_now[offset_now] = 0;
          v_scale_smem[offset_now] = 0;
        }
      }
      if (tile_start + offset_now + 8 >= start_len) {
        if (tile_start + offset_now + 8 < end_len) {
          cache_v_scale_now[offset_now + 8] =
              __hdiv(1, v_local_max_value[fz * 2 + 1]);
          v_scale_smem[offset_now + 8] = v_local_max_value[fz * 2 + 1];
        } else {
          cache_v_scale_now[offset_now + 8] = 0;
          v_scale_smem[offset_now + 8] = 0;
        }
      }
    }
    __syncthreads();
    k_smem_offset_r -= 2 * num_frags_y;  // num_frags_z = 1
  }
  __syncthreads();

  // mask, quant, store
  using LoadKVT = AlignedVector<uint8_t, 4>;
  LoadKVT cache_vec1;
  LoadKVT cache_vec2;

  uint32_t chunk_start_k = tile_start + wid * num_frags_z * 16 + tid / 4;
  uint32_t kv_frag[4];
  const uint32_t write_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_b_stride = HEAD_DIM;
  const uint32_t write_d_stride = BLOCK_SIZE;
  uint32_t k_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_z * 16 + tid / 4) * write_b_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
    uint32_t k_write_idx_now_z = k_write_idx + fz * 16 * write_b_stride;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t k_write_idx_now = k_write_idx_now_z +
                                 fy % 2 * 8 * write_b_stride +
                                 fy / 2 * 32;  // + fy % 2 * 16;
      // load
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // quant
      T *k_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_k + k_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_k + k_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_k + (v_id / 4) * 8 >= start_len &&
            chunk_start_k + (v_id / 4) * 8 < end_len) {
          uint_quant_value = QuantToC8<T, is_need_kv_quant, IsFP8>(
              k_local_max_value[fz * 2 + v_id / 4],
              k_frag_T[v_id],
              127.0f,
              -127.0f);
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id - 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_k + k_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_k + k_write_idx_now + 16);
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16, num_vecs_per_head>(k_smem_offset_r) -
        2 * num_frags_y;
    chunk_start_k += 16;
  }

  uint32_t chunk_start_v = tile_start + tid % 4 * 2;
  uint32_t v_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_v * 16 + tid / 4) * write_d_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
  const uint32_t num_frags_z_v = num_frags_z * NUM_WARPS;
  T v_scales[num_frags_z_v * 4];
  for (int v_i = 0; v_i < num_frags_z_v; v_i++) {
    const int offset = v_i * 16;
    const int t_offset = tid % 4 * 2;
    v_scales[v_i * 4] = v_scale_smem[offset + t_offset];
    v_scales[v_i * 4 + 1] = v_scale_smem[offset + t_offset + 1];
    v_scales[v_i * 4 + 2] = v_scale_smem[offset + t_offset + 8];
    v_scales[v_i * 4 + 3] = v_scale_smem[offset + t_offset + 9];
  }

#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_v; ++fy) {
    uint32_t v_write_idx_now_v = v_write_idx + fy * 16 * write_d_stride;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z_v; ++fz) {
      uint32_t v_write_idx_now = v_write_idx_now_v +
                                 fz % 2 * 8 * write_d_stride +
                                 fz / 2 * 32;  // + fz % 2 * 16;
      // load
      v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, kv_frag);
      // quant
      T *v_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_v + v_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_v + v_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 >= start_len &&
            chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 < end_len) {
          uint_quant_value = QuantToC8<T, is_need_kv_quant, IsFP8>(
              v_scales[fz * 4 + v_id % 4], v_frag_T[v_id], 127.0f, -127.0f);
          // store now
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id % 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_v + v_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_v + v_write_idx_now + 16);
      chunk_start_v += 16;
      v_smem_offset_r =
          k_smem.advance_offset_by_row<16, num_vecs_per_head>(v_smem_offset_r);
    }
    v_smem_offset_r = k_smem.advance_offset_by_column<2>(
                          v_smem_offset_r, wid * num_frags_v + fy) -
                      16 * num_frags_z_v * num_vecs_per_head;
    chunk_start_v -= 16 * num_frags_z_v;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

// Write Cache KV in Append
template <typename T,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS>
__global__ void append_write_cache_kv_c4_qkv(
    uint8_t *__restrict__ cache_k,
    uint8_t *__restrict__ cache_v,
    const T *__restrict__ qkv_input,
    const T *__restrict__ cache_k_scales,
    const T *__restrict__ cache_v_scales,
    const T *__restrict__ cache_k_zero_points,
    const T *__restrict__ cache_v_zero_points,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ batch_id_per_token,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ block_tables,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int kv_num_heads) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t pad_len = BLOCK_SIZE;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids[btid];
  const uint32_t seq_len_this_time = seq_lens_this_time[batch_id];
  if (seq_len_this_time <= 0) {
    return;
  }
  const int *block_table_now = nullptr;

  block_table_now = block_tables + batch_id * max_blocks_per_seq;

  const uint32_t num_rows_per_block =
      NUM_WARPS * num_frags_z * 16;  // BLOCK_SIZE
  const uint32_t start_len = seq_lens_decoder[batch_id];
  const uint32_t bf_pad_len = start_len % pad_len;
  const uint32_t start_len_pad = start_len - bf_pad_len;
  const uint32_t end_len = start_len + seq_len_this_time;
  const uint32_t tile_start = start_len_pad + tile_id * num_rows_per_block;
  uint32_t chunk_start = tile_start + wid * num_frags_z * 16 + tid / 8;

  const uint32_t start_token_idx = cu_seqlens_q[batch_id];
  const uint32_t kv_batch_stride = (num_heads + 2 * kv_num_heads) * HEAD_DIM;
  const uint32_t kv_h_stride = HEAD_DIM;
  int block_id = __ldg(&block_table_now[tile_start / BLOCK_SIZE]);

  const uint32_t HEAD_DIM_HALF = HEAD_DIM / 2;
  const uint32_t BLOCK_SIZE_HALF = BLOCK_SIZE / 2;

  if (tile_start >= start_len) {
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    // pad zero for this kv_head_idx for this block
    LoadPadKVT pad_cache_vec;
    *(reinterpret_cast<uint4 *>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
    // reset k
    constexpr int num_vecs_per_head_k = HEAD_DIM_HALF / KV_VEC_SIZE;  // 4
    constexpr int num_token_each_time_k = 32 / num_vecs_per_head_k;   // 8
    uint32_t tgt_idx =
        (block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE * HEAD_DIM_HALF +
        tid % num_vecs_per_head_k * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_k; block_i < BLOCK_SIZE;
         block_i += num_token_each_time_k) {
      Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                  &cache_k[tgt_idx + block_i * HEAD_DIM_HALF]);
    }

    // reset v
    const int num_vecs_per_head_v = BLOCK_SIZE_HALF / KV_VEC_SIZE;  // 2
    const int num_token_each_time_v = 32 / num_vecs_per_head_v;     // 16
    tgt_idx =
        (block_id * kv_num_heads + kv_head_idx) * HEAD_DIM * BLOCK_SIZE_HALF +
        tid % num_vecs_per_head_v * KV_VEC_SIZE;
    for (int block_i = tid / num_vecs_per_head_v; block_i < HEAD_DIM;
         block_i += num_token_each_time_v) {
      Store<uint8_t, KV_VEC_SIZE>(
          pad_cache_vec, &cache_v[tgt_idx + block_i * BLOCK_SIZE_HALF]);
    }
  }

  __shared__ T k_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T k_scale_smem[HEAD_DIM];
  __shared__ T v_scale_smem[HEAD_DIM];
  __shared__ T k_zero_point_smem[HEAD_DIM];
  __shared__ T v_zero_point_smem[HEAD_DIM];
  const T *cache_k_scale_now = cache_k_scales + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_points + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_scales + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_points + kv_head_idx * HEAD_DIM;
#pragma unroll
  for (uint32_t i = wid * 32 + tid; i < HEAD_DIM; i += 128) {
    k_scale_smem[i] = cache_k_scale_now[i];
    k_zero_point_smem[i] = cache_k_zp_now[i];
    v_scale_smem[i] = cache_v_scale_now[i];
    v_zero_point_smem[i] = cache_v_zp_now[i];
  }

  smem_t k_smem(k_smem_ori);
  smem_t v_smem(v_smem_ori);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid / 8, tid % 8);  // 4 * 8 per warp

  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  constexpr uint32_t num_frags_v = num_frags_y / NUM_WARPS;
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16,
      wid * num_frags_v * 2 + tid / 16);  // wid * num_frags_v * 16 / 8

  // load kv gmem to smem
  const uint32_t real_start_token_idx = start_token_idx - bf_pad_len +
                                        tile_id * num_rows_per_block +
                                        wid * num_frags_z * 16 + tid / 8;
  uint32_t k_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
  uint32_t v_read_idx = real_start_token_idx * kv_batch_stride +
                        (num_heads + kv_num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y / 4;
           ++fy) {  // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        if (chunk_start >= start_len && chunk_start < end_len) {
          k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + k_read_idx, chunk_start < end_len);
          v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + v_read_idx, chunk_start < end_len);
        }
        kv_smem_offset_w =
            k_smem.advance_offset_by_column<8>(kv_smem_offset_w, fy);
        k_read_idx += 8 * num_elems_per_128b<T>();
        v_read_idx += 8 * num_elems_per_128b<T>();
      }
      kv_smem_offset_w =
          k_smem.advance_offset_by_row<4, num_vecs_per_head>(kv_smem_offset_w) -
          2 * num_frags_y;
      k_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      v_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      chunk_start += 4;
    }
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // mask, quant, store
  T cache_k_scale_frag[num_frags_y][4];
  T cache_k_zp_frag[num_frags_y][4];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&k_scale_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&k_scale_smem[fy * 16]) + tid % 4 + 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&k_zero_point_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&k_zero_point_smem[fy * 16]) + tid % 4 +
          4);
  }
  T cache_v_scale_frag[num_frags_y][2];
  T cache_v_zp_frag[num_frags_y][2];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    cache_v_scale_frag[fy][0] = v_scale_smem[fy * 16 + tid / 4];
    cache_v_scale_frag[fy][1] = v_scale_smem[fy * 16 + tid / 4 + 8];
    cache_v_zp_frag[fy][0] = v_zero_point_smem[fy * 16 + tid / 4];
    cache_v_zp_frag[fy][1] = v_zero_point_smem[fy * 16 + tid / 4 + 8];
  }

  using LoadKVT = AlignedVector<uint8_t, 4>;
  LoadKVT cache_vec;

  uint32_t chunk_start_k = tile_start + wid * num_frags_z * 16 + tid / 4;
  uint32_t kv_frag[4];
  const uint32_t write_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t write_h_stride = BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t write_b_stride = HEAD_DIM / 2;
  const uint32_t write_d_stride = BLOCK_SIZE / 2;
  uint32_t k_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_z * 16 + tid / 4) * write_b_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
    uint32_t k_write_idx_now_z = k_write_idx + fz * 16 * write_b_stride;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t k_write_idx_now = k_write_idx_now_z +
                                 (fy % 4) / 2 * 8 * write_b_stride +
                                 fy / 4 * 32 + fy % 2 * 16;
      // load
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // quant
      T *k_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_k + k_write_idx_now, &cache_vec);
      }

#pragma unroll
      for (uint32_t v_id = 0; v_id < 4; ++v_id) {
        float quant_value1, quant_value2;
        uint8_t uint_quant_value1, uint_quant_value2;
        if (chunk_start_k >= start_len && chunk_start_k < end_len) {
          quant_value1 =
              static_cast<float>(cache_k_scale_frag[fy][v_id] * k_frag_T[v_id] +
                                 cache_k_zp_frag[fy][v_id]);
          quant_value1 = roundWithTiesToEven(quant_value1);
          quant_value1 = quant_value1 > 7.0f ? 7.0f : quant_value1;
          quant_value1 = quant_value1 < -8.0f ? -8.0f : quant_value1;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
        } else {
          uint_quant_value1 = 0;
        }
        if (chunk_start_k + 8 >= start_len && chunk_start_k + 8 < end_len) {
          quant_value2 = static_cast<float>(cache_k_scale_frag[fy][v_id] *
                                                k_frag_T[v_id + 4] +
                                            cache_k_zp_frag[fy][v_id]);
          quant_value2 = roundWithTiesToEven(quant_value2);
          quant_value2 = quant_value2 > 7.0f ? 7.0f : quant_value2;
          quant_value2 = quant_value2 < -8.0f ? -8.0f : quant_value2;
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
        } else {
          uint_quant_value2 = 0;
        }
        if (bf_pad_len != 0) {
          cache_vec[v_id] |=
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        } else {
          cache_vec[v_id] =
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec, cache_k + k_write_idx_now);
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16, num_vecs_per_head>(k_smem_offset_r) -
        2 * num_frags_y;
    chunk_start_k += 16;
  }

  uint32_t chunk_start_v = tile_start + tid % 4 * 2;
  uint32_t v_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_v * 16 + tid / 4) * write_d_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
  const uint32_t num_frags_z_v = num_frags_z * NUM_WARPS;
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_v; ++fy) {
    uint32_t v_write_idx_now_v = v_write_idx + fy * 16 * write_d_stride;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z_v; ++fz) {
      uint32_t v_write_idx_now = v_write_idx_now_v +
                                 (fz % 4) / 2 * 8 * write_d_stride +
                                 fz / 4 * 32 + fz % 2 * 16;
      // load
      v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, kv_frag);
      // quant
      T *v_frag_T = reinterpret_cast<T *>(kv_frag);

      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_v + v_write_idx_now, &cache_vec);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 4; ++v_id) {
        float quant_value1, quant_value2;
        uint8_t uint_quant_value1, uint_quant_value2;
        if (chunk_start_v + v_id % 2 + v_id / 2 * 8 >= start_len &&
            chunk_start_v + v_id % 2 + v_id / 2 * 8 < end_len) {
          quant_value1 = static_cast<float>(
              cache_v_scale_frag[wid * num_frags_v + fy][0] * v_frag_T[v_id] +
              cache_v_zp_frag[wid * num_frags_v + fy][0]);
          quant_value1 = roundWithTiesToEven(quant_value1);
          quant_value1 = quant_value1 > 7.0f ? 7.0f : quant_value1;
          quant_value1 = quant_value1 < -8.0f ? -8.0f : quant_value1;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          quant_value2 =
              static_cast<float>(cache_v_scale_frag[wid * num_frags_v + fy][1] *
                                     v_frag_T[v_id + 4] +
                                 cache_v_zp_frag[wid * num_frags_v + fy][1]);
          quant_value2 = roundWithTiesToEven(quant_value2);
          quant_value2 = quant_value2 > 7.0f ? 7.0f : quant_value2;
          quant_value2 = quant_value2 < -8.0f ? -8.0f : quant_value2;
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
        } else {
          uint_quant_value1 = 0;
          uint_quant_value2 = 0;
        }

        if (bf_pad_len != 0) {
          cache_vec[v_id] |=
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        } else {
          cache_vec[v_id] =
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec, cache_v + v_write_idx_now);
      chunk_start_v += 16;
      v_smem_offset_r =
          v_smem.advance_offset_by_row<16, num_vecs_per_head>(v_smem_offset_r);
    }
    v_smem_offset_r = v_smem.advance_offset_by_column<2>(
                          v_smem_offset_r, wid * num_frags_v + fy) -
                      16 * num_frags_z_v * num_vecs_per_head;
    chunk_start_v -= 16 * num_frags_z_v;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
}

template <typename T, typename QKV_TYPE>
void rotary_qk_variable(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    const QKV_TYPE *qkv_input,    // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const cudaStream_t &stream,
    bool use_neox_style = false,
    bool rope_3d = false) {
  int64_t elem_nums = qkv_out_scales ? token_num * 3 * head_num * dim_head
                                     : token_num * 2 * head_num * dim_head;
  if (use_neox_style) {
    elem_nums /= 2;
  }

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    if (qkv_out_scales) {
      launchWithPdlWhenEnabled(IntVariableLengthRotaryKernel<T, PackSize>,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const int *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out_scales,
                               qkv_bias,
                               qkv_out,
                               elem_nums,
                               head_num,
                               seq_len,
                               dim_head,
                               rope_3d);
    } else {
      launchWithPdlWhenEnabled(VariableLengthRotaryKernel<T, PackSize>,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const T *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out,
                               elem_nums,
                               head_num,
                               seq_len,
                               dim_head,
                               rope_3d);
    }
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    if (qkv_out_scales) {
      launchWithPdlWhenEnabled(IntNeoxVariableLengthRotaryKernel<T, PackSize>,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const int *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out_scales,
                               qkv_bias,
                               qkv_out,
                               elem_nums,
                               head_num,
                               seq_len,
                               dim_head,
                               rope_3d);
    } else {
      launchWithPdlWhenEnabled(NeoxVariableLengthRotaryKernel<T, PackSize>,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const T *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out,
                               elem_nums,
                               head_num,
                               seq_len,
                               dim_head,
                               rope_3d);
    }
  }
}

template <typename T, typename QKV_TYPE>
void gqa_rotary_qk_norm_variable(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    const QKV_TYPE *qkv_input,    // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const cudaStream_t &stream,
    bool use_neox_style = false,
    bool rope_3d = false,
    const float *q_norm_weight = nullptr,
    const float *k_norm_weight = nullptr,
    const float rms_norm_eps = 1e-6) {
  int64_t elem_nums =
      qkv_out_scales
          ? token_num * (num_heads + 2 * kv_num_heads) * dim_head
          : token_num * (num_heads + kv_num_heads) * dim_head;  // for all q k v
  if (dim_head != 128) {
    PADDLE_THROW(
        "gqa rotary with qk norm only support head_dim=128, but got %d.",
        dim_head);
  }
  constexpr int HEAD_DIM = 128;
  constexpr int PackSize = HEAD_DIM / kWarpSize;
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  dim3 Block_Size(kWarpSize, blocksize / kWarpSize, 1);

  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  launchWithPdlWhenEnabled(GQAVariableLengthRotaryQKNormKernel<T, PackSize>,
                           grid_size,
                           Block_Size,
                           0,
                           stream,
                           reinterpret_cast<const T *>(qkv_input),
                           cos_emb,
                           sin_emb,
                           batch_id_per_token,
                           cu_seqlens_q,
                           seq_lens,
                           seq_lens_decoder,
                           qkv_out,
                           elem_nums,
                           num_heads,
                           kv_num_heads,
                           seq_len,
                           dim_head,
                           rope_3d,
                           q_norm_weight,
                           k_norm_weight,
                           rms_norm_eps);
}

template <typename T, typename QKV_TYPE>
void gqa_rotary_qk_variable(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    const QKV_TYPE *qkv_input,    // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int rotary_dim,
    const cudaStream_t &stream,
    bool use_neox_style = false,
    bool rope_3d = false) {
  int64_t elem_nums =
      qkv_out_scales
          ? token_num * (num_heads + 2 * kv_num_heads) * dim_head
          : token_num * (num_heads + kv_num_heads) * dim_head;  // for all q k v
  if (use_neox_style) {
    elem_nums /= 2;
  }

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    if (qkv_out_scales) {
      launchWithPdlWhenEnabled(IntGQAVariableLengthRotaryKernel<T, PackSize>,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const int *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out_scales,
                               qkv_bias,
                               qkv_out,
                               elem_nums,
                               num_heads,
                               kv_num_heads,
                               seq_len,
                               dim_head,
                               rope_3d);
    } else {
      auto *kernelFn = GQAVariableLengthRotaryKernel<T, PackSize>;
      launchWithPdlWhenEnabled(kernelFn,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               reinterpret_cast<const T *>(qkv_input),
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens,
                               seq_lens_decoder,
                               qkv_out,
                               elem_nums,
                               num_heads,
                               kv_num_heads,
                               seq_len,
                               dim_head,
                               rope_3d);
    }
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    if (qkv_out_scales) {
      launchWithPdlWhenEnabled(
          IntGQANeoxVariableLengthRotaryKernel<T, PackSize>,
          grid_size,
          blocksize,
          0,
          stream,
          reinterpret_cast<const int *>(qkv_input),
          cos_emb,
          sin_emb,
          batch_id_per_token,
          cu_seqlens_q,
          seq_lens,
          seq_lens_decoder,
          qkv_out_scales,
          qkv_bias,
          qkv_out,
          elem_nums,
          num_heads,
          kv_num_heads,
          seq_len,
          dim_head,
          rope_3d);
    } else {
      if (rotary_dim < dim_head) {
        PD_CHECK((rotary_dim / 2) % PackSize == 0);
        elem_nums =
            qkv_out_scales
                ? token_num * (num_heads + 2 * kv_num_heads) * rotary_dim
                : token_num * (num_heads + kv_num_heads) *
                      rotary_dim;  // for all q k v
        if (use_neox_style) {
          elem_nums /= 2;
        }
        const int pack_num_new = elem_nums / PackSize;
        GetNumBlocks<128>(pack_num_new, &grid_size);
        auto *kernelFn = GQANeoxVariableLengthPartialRotaryKernel<T, PackSize>;
        launchWithPdlWhenEnabled(kernelFn,
                                 grid_size,
                                 blocksize,
                                 0,
                                 stream,
                                 reinterpret_cast<const T *>(qkv_input),
                                 cos_emb,
                                 rotary_emb + input_output_len * rotary_dim / 2,
                                 batch_id_per_token,
                                 cu_seqlens_q,
                                 seq_lens,
                                 seq_lens_decoder,
                                 qkv_out_scales,
                                 qkv_bias,
                                 qkv_out,
                                 elem_nums,
                                 num_heads,
                                 kv_num_heads,
                                 seq_len,
                                 dim_head,
                                 rotary_dim,
                                 rope_3d);
      } else {
        auto *kernelFn = GQANeoxVariableLengthRotaryKernel<T, PackSize>;
        launchWithPdlWhenEnabled(kernelFn,
                                 grid_size,
                                 blocksize,
                                 0,
                                 stream,
                                 reinterpret_cast<const T *>(qkv_input),
                                 cos_emb,
                                 sin_emb,
                                 batch_id_per_token,
                                 cu_seqlens_q,
                                 seq_lens,
                                 seq_lens_decoder,
                                 qkv_out_scales,
                                 qkv_bias,
                                 qkv_out,
                                 elem_nums,
                                 num_heads,
                                 kv_num_heads,
                                 seq_len,
                                 dim_head,
                                 rope_3d);
      }
    }
  }
}

template <typename T, typename QKV_TYPE>
void gqa_rotary_qk_quant_variable(
    T *qkv_out,                   // [token_num, 3, num_head, dim_head]
    const QKV_TYPE *qkv_input,    // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const T *cache_k_scales,
    const T *cache_v_scales,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const cudaStream_t &stream,
    bool use_neox_style = false,
    bool rope_3d = false) {
  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * dim_head;
  if (use_neox_style) {
    elem_nums /= 2;
  }

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  if (!use_neox_style) {
    if (qkv_out_scales) {
      launchWithPdlWhenEnabled(
          IntGQAVariableLengthRotaryQuantKVKernel<T, PackSize>,
          grid_size,
          blocksize,
          0,
          stream,
          reinterpret_cast<const int *>(qkv_input),
          cos_emb,
          sin_emb,
          qkv_out_scales,
          batch_id_per_token,
          cu_seqlens_q,
          seq_lens,
          seq_lens_decoder,
          qkv_bias,
          cache_k_scales,
          cache_v_scales,
          qkv_out,
          elem_nums,
          num_heads,
          kv_num_heads,
          seq_len,
          dim_head,
          rope_3d);
    } else {
      launchWithPdlWhenEnabled(
          GQAVariableLengthRotaryQuantKVKernel<T, PackSize>,
          grid_size,
          blocksize,
          0,
          stream,
          reinterpret_cast<const T *>(qkv_input),
          cos_emb,
          sin_emb,
          batch_id_per_token,
          cu_seqlens_q,
          seq_lens,
          seq_lens_decoder,
          qkv_bias,
          cache_k_scales,
          cache_v_scales,
          qkv_out,
          elem_nums,
          num_heads,
          kv_num_heads,
          seq_len,
          dim_head,
          rope_3d);
    }
  } else {
    PADDLE_THROW("Use_neox_style mode isn't implemented yet");
  }
}

template <typename T>
void CascadeAppendWriteCacheKVQKV(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor
        &qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
               // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const int max_seq_len,
    cudaStream_t &stream,
    paddle::Tensor *key_cache_out,
    paddle::Tensor *value_cache_out) {
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto num_tokens = meta_data.token_nums;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto head_dim = meta_data.head_dims;
  auto block_size = meta_data.block_size;

  const uint32_t elem_nums = num_tokens * 2 * kv_num_heads * head_dim;
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  launchWithPdlWhenEnabled(
      cache_kernel<T, PackSize>,
      grid_size,
      blocksize,
      0,
      stream,
      reinterpret_cast<T *>(const_cast<T *>(qkv.data<T>())),
      reinterpret_cast<T *>(key_cache_out->data<T>()),
      reinterpret_cast<T *>(value_cache_out->data<T>()),
      block_table.data<int>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      max_seq_len,
      max_blocks_per_seq,
      num_heads,
      head_dim,
      block_size,
      elem_nums,
      kv_num_heads);
}

template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
void CascadeAppendWriteCacheKVC8QKV(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor
        &cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor
        &cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::Tensor &qkv,            // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor &cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_ids,
    const paddle::Tensor &tile_ids_per_batch,
    int num_blocks_x_cpu,
    int max_seq_len,
    bool is_scale_channel_wise,
    const std::string &cache_quant_type,
    cudaStream_t &stream,
    paddle::Tensor *cache_k_out,
    paddle::Tensor *cache_v_out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto num_tokens = meta_data.token_nums;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto head_dim = meta_data.head_dims;

  const uint32_t pad_len = BLOCK_SIZE;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / num_warps;
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_row_per_block = num_warps * num_frags_z * 16;

  dim3 grids(num_blocks_x_cpu, 1, kv_num_heads);
  dim3 blocks(32, num_warps);

  const uint32_t smem_size = (BLOCK_SIZE * HEAD_DIM) * sizeof(T) * 2;
  if (cache_quant_type != "block_wise_fp8") {
    auto kernel_fn = append_write_cache_kv_c8_qkv<T,
                                                  num_frags_y,
                                                  num_frags_z,
                                                  HEAD_DIM,
                                                  BLOCK_SIZE,
                                                  num_warps,
                                                  true,
                                                  false>;
    if (cache_quant_type == "cache_fp8") {
      kernel_fn = append_write_cache_kv_c8_qkv<T,
                                               num_frags_y,
                                               num_frags_z,
                                               HEAD_DIM,
                                               BLOCK_SIZE,
                                               num_warps,
                                               true,
                                               true>;
    }
    if (is_scale_channel_wise) {
      kernel_fn = append_write_cache_kv_c8_qkv<T,
                                               num_frags_y,
                                               num_frags_z,
                                               HEAD_DIM,
                                               BLOCK_SIZE,
                                               num_warps,
                                               false>;
    }
    cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    launchWithPdlWhenEnabled(kernel_fn,
                             grids,
                             blocks,
                             0,
                             stream,
                             cache_k_out->data<uint8_t>(),
                             cache_v_out->data<uint8_t>(),
                             qkv.data<T>(),
                             cache_k_scale.data<T>(),
                             cache_v_scale.data<T>(),
                             batch_ids.data<int>(),
                             tile_ids_per_batch.data<int>(),
                             seq_lens_this_time.data<int>(),
                             seq_lens_decoder.data<int>(),
                             batch_id_per_token.data<int>(),
                             cu_seqlens_q.data<int>(),
                             block_table.data<int>(),
                             max_seq_len,
                             max_blocks_per_seq,
                             num_heads,
                             kv_num_heads);
  } else {
    auto kernel_fn = append_write_cache_kv_c8_qkv_dynamic<NV_TYPE,
                                                          num_frags_y,
                                                          num_frags_z,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          num_warps,
                                                          true,
                                                          true>;
    cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    launchWithPdlWhenEnabled(
        kernel_fn,
        grids,
        blocks,
        0,
        stream,
        cache_k_out->data<uint8_t>(),
        cache_v_out->data<uint8_t>(),
        reinterpret_cast<const NV_TYPE *>(qkv.data<T>()),
        const_cast<NV_TYPE *>(
            reinterpret_cast<const NV_TYPE *>(cache_k_scale.data<T>())),
        const_cast<NV_TYPE *>(
            reinterpret_cast<const NV_TYPE *>(cache_v_scale.data<T>())),
        batch_ids.data<int>(),
        tile_ids_per_batch.data<int>(),
        seq_lens_this_time.data<int>(),
        seq_lens_decoder.data<int>(),
        batch_id_per_token.data<int>(),
        cu_seqlens_q.data<int>(),
        block_table.data<int>(),
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads);
  }
}

template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
void CascadeAppendWriteCacheKVC4QKV(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor
        &cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor
        &cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::Tensor &qkv,            // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor &cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor &cache_k_zp,     // [num_kv_heads, head_dim]
    const paddle::Tensor &cache_v_zp,     // [num_kv_heads, head_dim]
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_ids,
    const paddle::Tensor &tile_ids_per_batch,
    int num_blocks_x_cpu,
    int max_seq_len,
    cudaStream_t &stream,
    paddle::Tensor *cache_k_out,
    paddle::Tensor *cache_v_out) {
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto num_tokens = meta_data.token_nums;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto head_dim = meta_data.head_dims;

  const uint32_t pad_len = BLOCK_SIZE;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / num_warps;
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_row_per_block = num_warps * num_frags_z * 16;

  dim3 grids(num_blocks_x_cpu, 1, kv_num_heads);
  dim3 blocks(32, num_warps);

  const uint32_t smem_size =
      (BLOCK_SIZE * HEAD_DIM) * sizeof(T) * 2 + HEAD_DIM * 4 * sizeof(T);
  auto kernel_fn = append_write_cache_kv_c4_qkv<T,
                                                num_frags_y,
                                                num_frags_z,
                                                HEAD_DIM,
                                                BLOCK_SIZE,
                                                num_warps>;
  cudaFuncSetAttribute(
      kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  launchWithPdlWhenEnabled(kernel_fn,
                           grids,
                           blocks,
                           0,
                           stream,
                           cache_k_out->data<uint8_t>(),
                           cache_v_out->data<uint8_t>(),
                           qkv.data<T>(),
                           cache_k_scale.data<T>(),
                           cache_v_scale.data<T>(),
                           cache_k_zp.data<T>(),
                           cache_v_zp.data<T>(),
                           batch_ids.data<int>(),
                           tile_ids_per_batch.data<int>(),
                           seq_lens_this_time.data<int>(),
                           seq_lens_decoder.data<int>(),
                           batch_id_per_token.data<int>(),
                           cu_seqlens_q.data<int>(),
                           block_table.data<int>(),
                           max_seq_len,
                           max_blocks_per_seq,
                           num_heads,
                           kv_num_heads);
}
