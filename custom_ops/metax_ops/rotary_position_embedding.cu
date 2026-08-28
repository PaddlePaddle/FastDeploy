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
__global__ void GQAVariableLengthRotaryKernel(T* qkv,
                                              const float* cos_emb,
                                              const float* sin_emb,
                                              const int* batch_id_per_token,
                                              const int* cu_seqlens_q,
                                              const int* seq_lens_encoder,
                                              const int* seq_lens_decoder,
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
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] == 0) continue;
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
    Store<T, VecSize>(src_vec, &qkv[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(T* qkv,
                                                  const float* cos_emb,
                                                  const float* sin_emb,
                                                  const int* batch_id_per_token,
                                                  const int* cu_seqlens_q,
                                                  const int* seq_lens_encoder,
                                                  const int* seq_lens_decoder,
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
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] == 0) continue;
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
    Store<T, VecSize>(left_vec, &qkv[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv[base_idx_right]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthPartialRotaryKernel(
    T* qkv,
    const float* cos_emb,
    const float* sin_emb,
    const int* batch_id_per_token,
    const int* cu_seqlens_q,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
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
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_bi = batch_id_per_token[token_idx];
    if (ori_bi == -1) continue;
    if (seq_lens_encoder && seq_lens_encoder[ori_bi] == 0) continue;
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
    Store<T, VecSize>(left_vec, &qkv[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv[base_idx_right]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(T* qkv,
                            const float* rotary_emb,
                            const int* batch_id_per_token,
                            const int* cu_seqlens_q,
                            const int* seq_lens_encoder,
                            const int* seq_lens_decoder,
                            const int token_num,
                            const int num_heads,
                            const int kv_num_heads,
                            const int seq_len,
                            const int input_output_len,
                            const int dim_head,
                            const int rotary_dim,
                            const cudaStream_t& stream,
                            bool use_neox_style,
                            bool rope_3d) {
  int64_t elem_nums = token_num * (num_heads + kv_num_heads) * dim_head;
  if (use_neox_style) {
    elem_nums /= 2;
  }

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  if (!use_neox_style) {
    const float* cos_emb = rotary_emb;
    const float* sin_emb = rotary_emb + input_output_len * dim_head / 2;
    auto* kernelFn = GQAVariableLengthRotaryKernel<T, PackSize>;
    launchWithPdlWhenEnabled(kernelFn,
                             grid_size,
                             blocksize,
                             0,
                             stream,
                             qkv,
                             cos_emb,
                             sin_emb,
                             batch_id_per_token,
                             cu_seqlens_q,
                             seq_lens_encoder,
                             seq_lens_decoder,
                             elem_nums,
                             num_heads,
                             kv_num_heads,
                             seq_len,
                             dim_head,
                             rope_3d);
  } else {
    const float* cos_emb = rotary_emb;
    const float* sin_emb = rotary_emb + input_output_len * dim_head;
    if (rotary_dim < dim_head) {
      PD_CHECK((rotary_dim / 2) % PackSize == 0);
      elem_nums = token_num * (num_heads + kv_num_heads) * rotary_dim;
      if (use_neox_style) {
        elem_nums /= 2;
      }
      const int pack_num_new = elem_nums / PackSize;
      GetNumBlocks<128>(pack_num_new, &grid_size);
      auto* kernelFn = GQANeoxVariableLengthPartialRotaryKernel<T, PackSize>;
      launchWithPdlWhenEnabled(kernelFn,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               qkv,
                               cos_emb,
                               rotary_emb + input_output_len * rotary_dim / 2,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens_encoder,
                               seq_lens_decoder,
                               elem_nums,
                               num_heads,
                               kv_num_heads,
                               seq_len,
                               dim_head,
                               rotary_dim,
                               rope_3d);
    } else {
      auto* kernelFn = GQANeoxVariableLengthRotaryKernel<T, PackSize>;
      launchWithPdlWhenEnabled(kernelFn,
                               grid_size,
                               blocksize,
                               0,
                               stream,
                               qkv,
                               cos_emb,
                               sin_emb,
                               batch_id_per_token,
                               cu_seqlens_q,
                               seq_lens_encoder,
                               seq_lens_decoder,
                               elem_nums,
                               num_heads,
                               kv_num_heads,
                               seq_len,
                               dim_head,
                               rope_3d);
    }
  }
}

void RotaryPositionEmbedding(
    paddle::Tensor&
        qkv,  // [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]
    const paddle::optional<paddle::Tensor>&
        seq_lens_encoder,                      // [max_batch_size, 1]
    const paddle::Tensor& seq_lens_decoder,    // [max_batch_size, 1]
    const paddle::Tensor& batch_id_per_token,  // [num_tokens,]
    const paddle::Tensor& cu_seqlens_q,        // [batch_size + 1,]
    const paddle::Tensor& rotary_embs,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const int max_seq_len,
    const bool use_neox_style,
    const bool rope_3d) {
  if (qkv.dtype() != paddle::DataType::BFLOAT16) {
    PD_THROW("Only support qkv dtype of BF16");
  }
  using data_t = phi::dtype::bfloat16;

  const auto rotary_embs_shape = rotary_embs.shape();
  const auto num_tokens = qkv.shape()[0];
  const auto hidden_size = (num_heads + 2 * kv_num_heads) * head_dim;
  const auto rotary_dim = rotary_embs_shape[rotary_embs_shape.size() - 1] * 2;
  if (rotary_dim < head_dim) {
    if (!use_neox_style || num_heads == kv_num_heads) {
      PADDLE_THROW(
          phi::errors::Fatal("partial_rotary_factor < 1.0 only supports "
                             "use_neox_rotary_style=True"));
    }
  }

  gqa_rotary_qk_variable(
      qkv.data<data_t>(),
      rotary_embs.data<float>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      seq_lens_encoder ? seq_lens_encoder->data<int>() : nullptr,
      seq_lens_decoder.data<int>(),
      num_tokens,
      num_heads,
      kv_num_heads,
      max_seq_len,
      rope_3d ? rotary_embs_shape[3] : rotary_embs_shape[2],
      head_dim,
      rotary_dim,
      qkv.stream(),
      use_neox_style,
      rope_3d);
}

PD_BUILD_STATIC_OP(rotary_position_embedding)
    .Inputs({"qkv",
             paddle::Optional("seq_lens_encoder"),
             "seq_lens_decoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "rotary_embs"})
    .Outputs({"qkv_out"})
    .Attrs({"num_heads:int",
            "kv_num_heads:int",
            "head_dim:int",
            "max_seq_len:int",
            "use_neox_style:bool",
            "rope_3d:bool"})
    .SetInplaceMap({{"qkv", "qkv_out"}})
    .SetKernelFn(PD_KERNEL(RotaryPositionEmbedding));
