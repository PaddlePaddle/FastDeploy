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

#include "helper.h"
#include "paddle/extension.h"

template <typename T, int VecSize = 1>
__global__ void FusedNeoxRopeEmbeddingKernel(const T *__restrict__ qkv,
                                             const float *__restrict__ cos_emb,
                                             const float *__restrict__ sin_emb,
                                             T *__restrict__ q,
                                             T *__restrict__ k,
                                             T *__restrict__ v,
                                             const int64_t elem_cnt,
                                             const int num_head,
                                             const int last_dim) {
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
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size +
                              qkv_id * full_hidden_size + hi * last_dim +
                              h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;
    const int emb_idx = token_idx * last_dim + h_bias;
    const int base_split_idx_left =
        token_idx * full_hidden_size + hi * last_dim + h_bias;
    const int base_split_idx_right = base_split_idx_left + half_lastdim;

    // q,k,v output
    T *out_p = nullptr;
    if (qkv_id == 0) {
      out_p = q;
    } else if (qkv_id == 1) {
      out_p = k;
    } else {
      out_p = v;
    }

    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    // do rope
    if (qkv_id < 2) {
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float input_left = static_cast<float>(left_vec[i]);
        float input_right = static_cast<float>(right_vec[i]);
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);

        int cur_idx_1 = base_split_idx_left + i;
        int cur_idx_2 = base_split_idx_right + i;
      }
    }
    Store<T, VecSize>(left_vec, &out_p[base_split_idx_left]);
    Store<T, VecSize>(right_vec, &out_p[base_split_idx_right]);
  }
}

std::vector<paddle::Tensor> FusedNeoxRopeEmbedding(
    const paddle::Tensor &qkv,
    const paddle::Tensor &cos_emb,
    const paddle::Tensor &sin_emb,
    const int num_heads,
    const int head_dim) {
  typedef PDTraits<paddle::DataType::BFLOAT16> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto &qkv_dims = qkv.dims();
  const int token_num = qkv_dims.size() == 2 ? qkv_dims[0] : qkv_dims[1];

  auto stream = qkv.stream();
  paddle::Tensor q = GetEmptyTensor(
      {token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor k = GetEmptyTensor(
      {token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor v = GetEmptyTensor(
      {token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());

  int64_t elem_nums = token_num * num_heads * head_dim * 3 / 2;
  constexpr int PackSize = 4;
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  FusedNeoxRopeEmbeddingKernel<DataType_, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
          reinterpret_cast<const DataType_ *>(qkv.data<data_t>()),
          cos_emb.data<float>(),
          sin_emb.data<float>(),
          reinterpret_cast<DataType_ *>(q.data<data_t>()),
          reinterpret_cast<DataType_ *>(k.data<data_t>()),
          reinterpret_cast<DataType_ *>(v.data<data_t>()),
          elem_nums,
          num_heads,
          head_dim);
  return {q, k, v};
}

PD_BUILD_STATIC_OP(fused_neox_rope_embedding)
    .Inputs({"qkv", "cos_emb", "sin_emb"})
    .Outputs({"q", "k", "v"})
    .Attrs({"num_heads: int", "head_dim: int"})
    .SetKernelFn(PD_KERNEL(FusedNeoxRopeEmbedding));
