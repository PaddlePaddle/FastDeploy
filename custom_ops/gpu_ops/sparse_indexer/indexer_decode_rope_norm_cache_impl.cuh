#pragma once

#include "helper.h"
#include "append_attn/mem_util.cuh"
#include "append_attn/mma_tensor_op.cuh"
#include "append_attn/utils.cuh"



template <typename T, int VecSize = 1> //4
__global__ void indexer_decode_cache_T_rope_qk_norm_kernel(
    const T* __restrict__ quant_qkv,  // [bsz, q_num_head + 2 * kv_num_heads,
                                      // head_dim]
    T* __restrict__ key_cache,        // [num_blocks, kv_num_heads, block_size,
                                      // head_dim // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,  // [bsz, max_blocks_per_seq]

    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens_decoder,          // [bsz] decoder
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int q_num_head,
    const int head_dim,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads,
    const bool rope_3d,
    const float* q_norm_weight,
    const float* k_norm_weight,
    const float rms_norm_eps) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadBiasT out_vec;
  LoadKVT cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  LoadFloat tmp_vec;
  LoadFloat q_norm_vec, k_norm_vec;

  int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
  int64_t all_warp_num = gridDim.x * blockDim.y;
  int64_t all_head_dim = elem_cnt / head_dim;

  const int64_t hidden_size = (q_num_head +  kv_num_heads) * head_dim;
  const int half_head_dim = head_dim / 2;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int gloabl_hi = global_warp_idx; gloabl_hi < all_head_dim;
       gloabl_hi += all_warp_num) {
    int64_t linear_index = gloabl_hi * head_dim + threadIdx.x * VecSize;
    const int batch_id = linear_index / hidden_size;
    const int dimid_in_hidden = linear_index % hidden_size;
    const int head_id = dimid_in_hidden / head_dim;  // q + k + v
    const int dim_id = dimid_in_hidden % head_dim;
    const int start_token_idx = cu_seqlens_q[batch_id];
    if (seq_lens_encoder[batch_id] > 0) return;
    const int write_seq_id = seq_lens_decoder[batch_id];
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + batch_id * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx = //这个dim 在整个qkv tensor中的offset
        start_token_idx * hidden_size + head_id * head_dim + dim_id;

    const int bias_idx = head_id * head_dim + dim_id;
    Load<T, VecSize>(&quant_qkv[ori_idx], &src_vec);
    if (head_id < q_num_head + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_head_dim + dim_id / 2;
      uint32_t new_emb_idx =
          rope_3d ? emb_idx + batch_id * max_seq_len * head_dim : emb_idx;
      Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
    }
    float thread_m2 = 0.0f;
    float warp_m2 = 0.0f;

#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // dequant + add_bias + rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);

      if (head_id < q_num_head + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        float tmp1 = input_left * cos_tmp - input_right * sin_tmp;
        float tmp2 = input_right * cos_tmp + input_left * sin_tmp;
        thread_m2 += tmp1 * tmp1 + tmp2 * tmp2;
        tmp_vec[2 * i] = tmp1;
        tmp_vec[2 * i + 1] = tmp2;
      } else {
        out_vec[2 * i] = src_vec[2 * i];
        out_vec[2 * i + 1] = src_vec[2 * i + 1];
      }
    }
    if (head_id < (q_num_head + kv_num_heads)) {  // q k
      WelfordWarpAllReduce<float, 32>(thread_m2, &warp_m2);
      float row_variance = max(warp_m2 / head_dim, 0.0f);
      float row_inv_var = Rsqrt(row_variance + rms_norm_eps);
      if (head_id < q_num_head) {  // q
        Load<float, VecSize>(&q_norm_weight[threadIdx.x * VecSize],
                             &q_norm_vec);
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          out_vec[i] = static_cast<T>(tmp_vec[i] * row_inv_var * q_norm_vec[i]);
        }
      } else {  // k
        Load<float, VecSize>(&k_norm_weight[threadIdx.x * VecSize],
                             &k_norm_vec);
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          out_vec[i] = static_cast<T>(tmp_vec[i] * row_inv_var * k_norm_vec[i]);
        }
      }
    }
    if (head_id < q_num_head) {
      // write q
      Store<T, VecSize>(out_vec, &qkv_out[ori_idx]);
    } else {
      // quant + write k/v
      const uint32_t kv_head_idx = (head_id - q_num_head) % kv_num_heads;
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * head_dim +
          kv_head_idx * block_size * head_dim + block_offset * head_dim +
          dim_id;
      if (head_id < q_num_head + kv_num_heads) {
        Store<T, VecSize>(out_vec, &key_cache[tgt_idx]);
      }
    //   } else {
    //     Store<T, VecSize>(out_vec, &value_cache[tgt_idx]);
    //   }
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}