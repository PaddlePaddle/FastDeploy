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

#include "encoder_write_cache_with_rope_impl.cuh"
#include "helper.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "remote_cache_kv_ipc.h"

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotarySplitKernel(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int *cu_seqlens_k,
    T *qkv_out,
    T *q,
    T *k,
    T *v,
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
  const int offset = (q_num_head + kv_num_head * 2) * last_dim;
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
    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < q_num_head) {
      base_split_idx =
          token_idx * q_num_head * last_dim + hi * last_dim + h_bias;
      out_p = q;
    } else if (hi < q_num_head + kv_num_head) {
      base_split_idx = kv_write_idx * kv_num_head * last_dim +
                       (hi - q_num_head) * last_dim + h_bias;
      out_p = k;
    } else {
      out_p = v;
      base_split_idx = kv_write_idx * kv_num_head * last_dim +
                       (hi - q_num_head - kv_num_head) * last_dim + h_bias;
    }
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    // do rope
    if (hi < q_num_head + kv_num_head) {
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
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
    Store<T, VecSize>(src_vec, &out_p[base_split_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_split_variable(
    T *qkv_out,  // [token_num, 3, num_head, dim_head]
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const bool rope_3d,
    const cudaStream_t &stream) {
  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * dim_head;
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  launchWithPdlWhenEnabled(GQAVariableLengthRotarySplitKernel<T, PackSize>,
                           grid_size,
                           blocksize,
                           0,
                           stream,
                           qkv_input,
                           cos_emb,
                           sin_emb,
                           batch_id_per_token,
                           cu_seqlens_q,
                           seq_lens_encoder,
                           seq_lens_decoder,
                           cu_seqlens_k,
                           qkv_out,
                           q,
                           k,
                           v,
                           elem_nums,
                           num_heads,
                           kv_num_heads,
                           seq_len,
                           dim_head,
                           rope_3d);
}

template <typename T,
          typename CacheT,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS = 4>
__global__ void append_cache_kv_c16(const T *__restrict__ cache_k,
                                    const T *__restrict__ cache_v,
                                    T *__restrict__ k_out,
                                    T *__restrict__ v_out,
                                    const int *__restrict__ seq_lens_this_time,
                                    const int *__restrict__ seq_lens_decoder,
                                    const int *__restrict__ cu_seqlens_k,
                                    const int *__restrict__ block_tables,
                                    const int *batch_ids,
                                    const int *tile_ids_per_batch,
                                    const int max_blocks_per_seq,
                                    const int kv_num_heads) {
  // start_kv_idx: start kv_idx current block
  // batch_id：block's batch_id
  // TODO: 1.scale preload 2.frag_dq_T reuse 3.pipeline 4.store aligned 5.cacheT
  // with template（int8/fp8)
  const uint32_t tile_idx = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;

  const uint32_t batch_id = batch_ids[tile_idx];
  const uint32_t start_kv_idx = tile_ids_per_batch[tile_idx] * BLOCK_SIZE;
  const uint32_t end_idx = seq_lens_decoder[batch_id] - start_kv_idx;
  if (seq_lens_this_time[batch_id] <= 0) {
    return;
  }

  const int *cur_block_table = block_tables + batch_id * max_blocks_per_seq;
  uint32_t block_id = cur_block_table[start_kv_idx / BLOCK_SIZE];
  // cache_kv idx
  uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  uint32_t block_stride = kv_num_heads * kv_h_stride;
  const CacheT *cur_cache_k =
      cache_k + block_id * block_stride + kv_head_idx * kv_h_stride;
  const CacheT *cur_cache_v =
      cache_v + block_id * block_stride + kv_head_idx * kv_h_stride;

  // k_out v_out idx
  uint32_t kv_t_stride = kv_num_heads * HEAD_DIM;
  T *k_write_ptr =
      k_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
  T *v_write_ptr =
      v_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;

  uint32_t kv_frag[4];
  T *frag_dq_T = reinterpret_cast<T *>(kv_frag);

  constexpr uint32_t num_vecs_per_head =
      HEAD_DIM / num_elems_per_128b<CacheT>();
  constexpr uint32_t inv_kv_stride = 8 / num_vecs_per_head;

  extern __shared__ uint8_t smem[];
  smem_t k_smem(smem);
  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
          wid * 4 + tid / 8, tid % 8);  // 4 * 4 per warp

  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_read_idx =
      (wid * 4 + tid / 8) * HEAD_DIM + tid % 8 * num_elems_per_128b<CacheT>();

  // load k_smem 64 rows 128 cols
  for (int fz = 0; fz < 4;
       fz++) {  // 4 rows pre warp once, 16 rows all 4 warps once, need 4 iter
    for (int fy = 0; fy < 2; fy++) {  // 8 * 128b = 64 * bf16 once, need 2 iter
      k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          k_smem_offset_w, cur_cache_k + k_read_idx, end_idx > 0);
      k_smem_offset_w = k_smem.advance_offset_by_column<8, num_vecs_per_head>(
          k_smem_offset_w, fy);
      k_read_idx += 8 * num_elems_per_128b<CacheT>();
    }
    k_smem_offset_w =
        k_smem.advance_offset_by_row<4 * NUM_WARPS, num_vecs_per_head>(
            k_smem_offset_w) -
        16;
    k_read_idx += 4 * NUM_WARPS * HEAD_DIM - 16 * num_elems_per_128b<CacheT>();
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal k_smem 64 rows 128 cols
  for (int fz = 0; fz < 1;
       fz++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
    uint32_t row_idx = wid * 16 + tid / 4;
    for (int fy = 0; fy < 8; fy++) {  // 2 * 128b = 16 * bf16 once, need 8 iter
      uint32_t col_idx = fy * 16 + tid % 4 * 2;
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // layout
      /***
        r0c0,r0c1, r0c8,r0c9
        r8c0,r8c1, r8c8,r8c9
      ***/
      T *k_tile_ptr0 = k_write_ptr + row_idx * kv_t_stride +
                       kv_head_idx * HEAD_DIM + col_idx;
      T *k_tile_ptr1 = k_tile_ptr0 + 8 * kv_t_stride;

      if (row_idx < end_idx) {
        k_tile_ptr0[0] = frag_dq_T[0];
        k_tile_ptr0[1] = frag_dq_T[1];
        k_tile_ptr0[8] = frag_dq_T[2];
        k_tile_ptr0[9] = frag_dq_T[3];
      }

      if (row_idx + 8 < end_idx) {
        k_tile_ptr1[0] = frag_dq_T[4];
        k_tile_ptr1[1] = frag_dq_T[5];
        k_tile_ptr1[8] = frag_dq_T[6];
        k_tile_ptr1[9] = frag_dq_T[7];
      }
      k_smem_offset_r = k_smem.advance_offset_by_column<2, num_vecs_per_head>(
          k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head>(
            k_smem_offset_r) -
        16;
  }

  // ================v================
  smem_t v_smem(smem + BLOCK_SIZE * HEAD_DIM * sizeof(CacheT));
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
          wid * 4 + tid / 8, tid % 8);  // 4 * 4 per warp
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t v_read_idx =
      (wid * 4 + tid / 8) * HEAD_DIM + tid % 8 * num_elems_per_128b<CacheT>();

  // load v_smem 64 rows 128 cols
  for (int fz = 0; fz < 4; fz++) {    // // 4 rows pre warp once, 16 rows all 4
                                      // warps once, need 4 iter
    for (int fy = 0; fy < 2; fy++) {  // 8 * 128b = 64 * bf16 once, need 2 iter
      v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          v_smem_offset_w, cur_cache_v + v_read_idx, end_idx > 0);
      v_smem_offset_w = v_smem.advance_offset_by_column<8, num_vecs_per_head>(
          v_smem_offset_w, fy);
      v_read_idx += 8 * num_elems_per_128b<CacheT>();
    }
    v_smem_offset_w =
        v_smem.advance_offset_by_row<4 * NUM_WARPS, num_vecs_per_head>(
            v_smem_offset_w) -
        16;
    v_read_idx += 4 * NUM_WARPS * HEAD_DIM - 16 * num_elems_per_128b<CacheT>();
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal v_smem 64 rows 128 cols
  for (int fz = 0; fz < 1;
       fz++) {  //  16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
    uint32_t row_idx = wid * 16 + tid / 4;
    for (int fy = 0; fy < 8; fy++) {  // 2 * 128b = 16 * bf16 once, need 8 iter
      uint32_t col_idx = fy * 16 + tid % 4 * 2;
      v_smem.ldmatrix_m8n8x4(v_smem_offset_r, kv_frag);
      // layout
      /***
        r0c0,r0c1, r0c8,r0c9
        r8c0,r8c1, r8c8,r8c9
      ***/
      T *v_tile_ptr0 = v_write_ptr + row_idx * kv_t_stride +
                       kv_head_idx * HEAD_DIM + col_idx;
      T *v_tile_ptr1 = v_tile_ptr0 + 8 * kv_t_stride;

      if (row_idx < end_idx) {
        v_tile_ptr0[0] = frag_dq_T[0];
        v_tile_ptr0[1] = frag_dq_T[1];
        v_tile_ptr0[8] = frag_dq_T[2];
        v_tile_ptr0[9] = frag_dq_T[3];
      }

      if (row_idx + 8 < end_idx) {
        v_tile_ptr1[0] = frag_dq_T[4];
        v_tile_ptr1[1] = frag_dq_T[5];
        v_tile_ptr1[8] = frag_dq_T[6];
        v_tile_ptr1[9] = frag_dq_T[7];
      }
      v_smem_offset_r = v_smem.advance_offset_by_column<2, num_vecs_per_head>(
          v_smem_offset_r, fy);
    }
    v_smem_offset_r =
        v_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head>(
            v_smem_offset_r) -
        16;
  }
}

template <typename T,
          typename CacheT,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS = 4,
          bool IS_FP8 = false>
__global__ void append_cache_kv_c8(const CacheT *__restrict__ cache_k,
                                   const CacheT *__restrict__ cache_v,
                                   T *__restrict__ k_out,
                                   T *__restrict__ v_out,
                                   const T *__restrict__ cache_k_dequant_scales,
                                   const T *__restrict__ cache_v_dequant_scales,
                                   const int *__restrict__ seq_lens_this_time,
                                   const int *__restrict__ seq_lens_decoder,
                                   const int *__restrict__ cu_seqlens_k,
                                   const int *__restrict__ block_tables,
                                   const int *batch_ids,
                                   const int *tile_ids_per_batch,
                                   const int max_blocks_per_seq,
                                   const int kv_num_heads) {
  // start_kv_idx: start kv_idx current block
  // batch_id：block's batch_id
  // TODO: 1.scale preload 2.frag_dq_T reuse 3.pipeline 4.store aligned 5.cacheT
  // with template（int8/fp8)
  const uint32_t tile_idx = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;

  const uint32_t batch_id = batch_ids[tile_idx];
  const uint32_t start_kv_idx = tile_ids_per_batch[tile_idx] * BLOCK_SIZE;
  const uint32_t end_idx = seq_lens_decoder[batch_id] - start_kv_idx;
  if (seq_lens_this_time[batch_id] <= 0) {
    return;
  }

  const int *cur_block_table = block_tables + batch_id * max_blocks_per_seq;
  uint32_t block_id = cur_block_table[start_kv_idx / BLOCK_SIZE];
  // cache_kv idx
  uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  uint32_t block_stride = kv_num_heads * kv_h_stride;
  const CacheT *cur_cache_k =
      cache_k + block_id * block_stride + kv_head_idx * kv_h_stride;
  const CacheT *cur_cache_v =
      cache_v + block_id * block_stride + kv_head_idx * kv_h_stride;

  // k_out v_out idx
  uint32_t kv_t_stride = kv_num_heads * HEAD_DIM;
  T *k_write_ptr =
      k_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
  T *v_write_ptr =
      v_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;

  uint32_t k_frag[4], v_frag[4], frag_dq[4];
  T *frag_dq_T = reinterpret_cast<T *>(frag_dq);
  T cache_k_scale = cache_k_dequant_scales[kv_head_idx];
  T cache_v_scale = cache_v_dequant_scales[kv_head_idx];

  constexpr uint32_t num_vecs_per_head_k =
      HEAD_DIM / num_elems_per_128b<CacheT>();
  constexpr uint32_t num_vecs_per_blocksize =
      BLOCK_SIZE / num_elems_per_128b<CacheT>();
  constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
  constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;

  extern __shared__ uint8_t smem[];
  smem_t k_smem(smem);
  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 4 + tid / 8, tid % 8);  // 4 * 4 per warp

  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_read_idx =
      (wid * 4 + tid / 8) * HEAD_DIM + tid % 8 * num_elems_per_128b<CacheT>();

  // load v_smem 64 rows, 128 cols
  for (int fz = 0; fz < 4;
       fz++) {  // 4 rows pre warp once, 16 rows all 4 warps once, need 4 iter
    for (int fy = 0; fy < 1;
         fy++) {  // 8 * 128b = 128 * uint8 once, need 1 iter
      k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          k_smem_offset_w, cur_cache_k + k_read_idx, end_idx > 0);
      k_smem_offset_w = k_smem.advance_offset_by_column<8, num_vecs_per_head_k>(
          k_smem_offset_w, fy);
      k_read_idx += 8 * num_elems_per_128b<CacheT>();
    }
    k_smem_offset_w =
        k_smem.advance_offset_by_row<4 * NUM_WARPS, num_vecs_per_head_k>(
            k_smem_offset_w) -
        8;
    k_read_idx += 4 * NUM_WARPS * HEAD_DIM - 8 * num_elems_per_128b<CacheT>();
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal k_smem 64 rows, 128 cols
  for (int fz = 0; fz < 1;
       fz++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
    uint32_t row_idx = wid * 16 + tid / 4;
    for (int fy = 0; fy < 4; fy++) {  // 2 * 128b = 32 * uint8 once, need 4 iter
      uint32_t col_idx = fy * 32 + tid % 4 * 2;
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);
      // layout
      /***
      r0c0,r0c1,r0c8,r0c9, r8c0,r8c1,r8c8,r8c9
      r0c16,r0c17,r0c24,r0c25, r8c16,r8c17,r8c24,r8c25
      ***/
      for (int i = 0; i < 4 / 2; i++) {
        T *k_tile_ptr0 = k_write_ptr + row_idx * kv_t_stride +
                         kv_head_idx * HEAD_DIM + col_idx;
        T *k_tile_ptr1 = k_tile_ptr0 + 8 * kv_t_stride;

        if (row_idx < end_idx) {
          convert_c8<T, IS_FP8>(frag_dq_T,
                                k_frag[2 * i]);  // 4 * uint8/fp8 -> 4 * T
          k_tile_ptr0[0] = frag_dq_T[0] * cache_k_scale;
          k_tile_ptr0[1] = frag_dq_T[1] * cache_k_scale;
          k_tile_ptr0[8] = frag_dq_T[2] * cache_k_scale;
          k_tile_ptr0[9] = frag_dq_T[3] * cache_k_scale;
        }

        if (row_idx + 8 < end_idx) {
          convert_c8<T, IS_FP8>(frag_dq_T + 4,
                                k_frag[2 * i + 1]);  // 4 * uint8/fp8 -> 4 * T
          k_tile_ptr1[0] = frag_dq_T[4] * cache_k_scale;
          k_tile_ptr1[1] = frag_dq_T[5] * cache_k_scale;
          k_tile_ptr1[8] = frag_dq_T[6] * cache_k_scale;
          k_tile_ptr1[9] = frag_dq_T[7] * cache_k_scale;
        }
        col_idx += 16;
      }
      k_smem_offset_r = k_smem.advance_offset_by_column<2, num_vecs_per_head_k>(
          k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head_k>(
            k_smem_offset_r) -
        8;
  }

  // ================v================
  smem_t v_smem(smem + BLOCK_SIZE * HEAD_DIM * sizeof(CacheT));
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 8 + tid / 4, tid % 4);  // 4 * 8 per warp

  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t v_read_idx =
      (wid * 8 + tid / 4) * BLOCK_SIZE + tid % 4 * num_elems_per_128b<CacheT>();
  // load v_smem 128 rows 64 cols
  for (int fy = 0; fy < 4;
       fy++) {  // 8 rows pre warp once, 32 rows all 4 warps once, need 4 iter
    for (int fz = 0; fz < 1; fz++) {  // 4 * 128b = 64 * uint8 once, need 1 iter
      v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          v_smem_offset_w, cur_cache_v + v_read_idx, end_idx > 0);
      v_smem_offset_w =
          v_smem.advance_offset_by_column<4, num_vecs_per_blocksize>(
              v_smem_offset_w, fz);
      v_read_idx += 4 * num_elems_per_128b<CacheT>();
    }
    v_smem_offset_w =
        v_smem.advance_offset_by_row<8 * NUM_WARPS, num_vecs_per_blocksize>(
            v_smem_offset_w) -
        4;
    v_read_idx += 8 * NUM_WARPS * BLOCK_SIZE - 4 * num_elems_per_128b<CacheT>();
  }

  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal v_smem 128 rows 64 cols
  for (int fy = 0; fy < 2;
       fy++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 2 iter
    uint32_t dim_idx = fy * NUM_WARPS * 16 + wid * 16 + tid / 4;
    for (int fz = 0; fz < 2; fz++) {  // 2 * 128b = 32 * uint8 once, need 2 iter
      uint32_t kv_idx = fz * 32 + tid % 4 * 2;
      v_smem.ldmatrix_m8n8x4(v_smem_offset_r, v_frag);
      // layout
      for (int i = 0; i < 4 / 2; i++) {
        T *v_tile_ptr0 = v_write_ptr + kv_idx * kv_t_stride +
                         kv_head_idx * HEAD_DIM + dim_idx;
        T *v_tile_ptr1 = v_tile_ptr0 + 8;
        convert_c8<T, IS_FP8>(frag_dq_T,
                              v_frag[2 * i]);  // 4 * uint8/fp8 -> 4 * T
        convert_c8<T, IS_FP8>(frag_dq_T + 4,
                              v_frag[2 * i + 1]);  // 4 * uint8/fp8 -> 4 * T
        if (kv_idx < end_idx) {
          v_tile_ptr0[0] = frag_dq_T[0] * cache_v_scale;
          v_tile_ptr1[0] = frag_dq_T[4] * cache_v_scale;
        }
        if (kv_idx + 1 < end_idx) {
          v_tile_ptr0[kv_t_stride] = frag_dq_T[1] * cache_v_scale;
          v_tile_ptr1[kv_t_stride] = frag_dq_T[5] * cache_v_scale;
        }
        if (kv_idx + 8 < end_idx) {
          v_tile_ptr0[8 * kv_t_stride] = frag_dq_T[2] * cache_v_scale;
          v_tile_ptr1[8 * kv_t_stride] = frag_dq_T[6] * cache_v_scale;
        }
        if (kv_idx + 9 < end_idx) {
          v_tile_ptr0[9 * kv_t_stride] = frag_dq_T[3] * cache_v_scale;
          v_tile_ptr1[9 * kv_t_stride] = frag_dq_T[7] * cache_v_scale;
        }
        kv_idx += 16;
      }
      v_smem_offset_r =
          v_smem.advance_offset_by_column<2, num_vecs_per_blocksize>(
              v_smem_offset_r, fz);
    }
    v_smem_offset_r =
        v_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_blocksize>(
            v_smem_offset_r) -
        4;
  }
}

template <typename T,
          typename CacheT,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS = 4>
__global__ void append_cache_kv_c4(const CacheT *__restrict__ cache_k,
                                   const CacheT *__restrict__ cache_v,
                                   T *__restrict__ k_out,
                                   T *__restrict__ v_out,
                                   const T *__restrict__ cache_k_dequant_scales,
                                   const T *__restrict__ cache_v_dequant_scales,
                                   const T *__restrict__ cache_k_zero_point,
                                   const T *__restrict__ cache_v_zero_point,
                                   const int *__restrict__ seq_lens_this_time,
                                   const int *__restrict__ seq_lens_decoder,
                                   const int *__restrict__ cu_seqlens_k,
                                   const int *__restrict__ block_tables,
                                   const int *batch_ids,
                                   const int *tile_ids_per_batch,
                                   const int max_blocks_per_seq,
                                   const int kv_num_heads) {
  // start_kv_idx: start kv_idx current block
  // batch_id：block's batch_id
  // TODO: 1.scale preload 2.frag_dq_T reuse 3.pipeline 4.store aligned 5.cacheT
  // with template（int8/fp8)
  const uint32_t tile_idx = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;

  const uint32_t batch_id = batch_ids[tile_idx];
  const uint32_t start_kv_idx = tile_ids_per_batch[tile_idx] * BLOCK_SIZE;
  const uint32_t end_idx = seq_lens_decoder[batch_id] - start_kv_idx;
  if (seq_lens_this_time[batch_id] <= 0) {
    return;
  }

  const int *cur_block_table = block_tables + batch_id * max_blocks_per_seq;
  uint32_t block_id = cur_block_table[start_kv_idx / BLOCK_SIZE];
  if (block_id < 0) block_id = 0;

  constexpr uint32_t HEAD_DIM_HALF = HEAD_DIM / 2;
  constexpr uint32_t BLOCK_SIZE_HALF = BLOCK_SIZE / 2;
  // cache_kv idx
  uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM_HALF;
  uint32_t block_stride = kv_num_heads * kv_h_stride;
  const CacheT *cur_cache_k =
      cache_k + block_id * block_stride + kv_head_idx * kv_h_stride;
  const CacheT *cur_cache_v =
      cache_v + block_id * block_stride + kv_head_idx * kv_h_stride;

  // k_out v_out idx
  uint32_t kv_t_stride = kv_num_heads * HEAD_DIM;
  T *k_write_ptr =
      k_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
  T *v_write_ptr =
      v_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;

  extern __shared__ uint8_t smem[];

  uint32_t k_frag[4], v_frag[4], frag_dq[8];
  T *frag_dq_T = reinterpret_cast<T *>(frag_dq);

  // load dequant scales and zero points
  const T *cache_k_scale_now = cache_k_dequant_scales + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_point + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_dequant_scales + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_point + kv_head_idx * HEAD_DIM;
  T *cache_k_scale_smem =
      reinterpret_cast<T *>(smem + BLOCK_SIZE * HEAD_DIM * sizeof(CacheT));
  T *cache_k_zero_point_smem = cache_k_scale_smem + HEAD_DIM;
  T *cache_v_scale_smem = cache_k_zero_point_smem + HEAD_DIM;
  T *cache_v_zero_point_smem = cache_v_scale_smem + HEAD_DIM;
#pragma unroll
  for (uint32_t i = wid * 32 + tid; i < HEAD_DIM; i += 128) {
    cache_k_scale_smem[i] = cache_k_scale_now[i];
    cache_k_zero_point_smem[i] = cache_k_zp_now[i] + static_cast<T>(136.f);
    cache_v_scale_smem[i] = cache_v_scale_now[i];
    cache_v_zero_point_smem[i] = cache_v_zp_now[i] + static_cast<T>(136.f);
  }

  smem_t k_smem(smem);
  constexpr uint32_t num_vecs_per_head_k =
      HEAD_DIM_HALF / num_elems_per_128b<CacheT>();  // 2
  constexpr uint32_t num_vecs_per_blocksize =
      BLOCK_SIZE_HALF / num_elems_per_128b<CacheT>();
  constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;  // 4
  constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 8 + tid / 4, tid % 4);  // 2(iter) * 4(warp) * 8 row per warp

  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);  //

  uint32_t k_read_idx = (wid * 8 + tid / 4) * HEAD_DIM / 2 +
                        tid % 4 * num_elems_per_128b<CacheT>();

  // load k_smem 64 rows 128 cols
  for (int fz = 0; fz < 2;
       fz++) {  // 4 rows pre warp once, 16 rows all 4 warps once, need 4 iter
    for (int fy = 0; fy < 1; fy++) {  // 4 * 128b = 128 * int4 once, need 1 iter
      k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          k_smem_offset_w, cur_cache_k + k_read_idx, end_idx > 0);
      k_smem_offset_w = k_smem.advance_offset_by_column<4, num_vecs_per_head_k>(
          k_smem_offset_w, fy);
      k_read_idx += 4 * num_elems_per_128b<CacheT>();
    }
    k_smem_offset_w =
        k_smem.advance_offset_by_row<8 * NUM_WARPS, num_vecs_per_head_k>(
            k_smem_offset_w) -
        4;
    k_read_idx +=
        8 * NUM_WARPS * HEAD_DIM / 2 - 4 * num_elems_per_128b<CacheT>();
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal k_smem 64 rows 128 cols
  for (int fz = 0; fz < 1;
       fz++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
    uint32_t row_idx = wid * 16 + tid / 4;
    for (int fy = 0; fy < 2; fy++) {  // 2 * 128b = 64 * int4 once, need 2 iter
      uint32_t col_idx = fy * 64 + tid % 4 * 2;
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);

      for (int i = 0; i < 2; i++) {
        T *k_tile_ptr0 = k_write_ptr + row_idx * kv_t_stride +
                         kv_head_idx * HEAD_DIM + col_idx;
        T *k_tile_ptr1 = k_tile_ptr0 + 8 * kv_t_stride;
        convert_int4(frag_dq_T, k_frag[2 * i]);
        convert_int4(frag_dq_T + 8, k_frag[2 * i + 1]);

        if (row_idx < end_idx) {
          k_tile_ptr0[0] = (frag_dq_T[0] - cache_k_zero_point_smem[col_idx]) *
                           cache_k_scale_smem[col_idx];
          k_tile_ptr0[1] =
              (frag_dq_T[1] - cache_k_zero_point_smem[col_idx + 1]) *
              cache_k_scale_smem[col_idx + 1];
          k_tile_ptr0[8] =
              (frag_dq_T[2] - cache_k_zero_point_smem[col_idx + 8]) *
              cache_k_scale_smem[col_idx + 8];
          k_tile_ptr0[9] =
              (frag_dq_T[3] - cache_k_zero_point_smem[col_idx + 9]) *
              cache_k_scale_smem[col_idx + 9];
          k_tile_ptr0[16] =
              (frag_dq_T[8] - cache_k_zero_point_smem[col_idx + 16]) *
              cache_k_scale_smem[col_idx + 16];
          k_tile_ptr0[17] =
              (frag_dq_T[9] - cache_k_zero_point_smem[col_idx + 17]) *
              cache_k_scale_smem[col_idx + 17];
          k_tile_ptr0[24] =
              (frag_dq_T[10] - cache_k_zero_point_smem[col_idx + 24]) *
              cache_k_scale_smem[col_idx + 24];
          k_tile_ptr0[25] =
              (frag_dq_T[11] - cache_k_zero_point_smem[col_idx + 25]) *
              cache_k_scale_smem[col_idx + 25];
        }

        if (row_idx + 8 < end_idx) {
          k_tile_ptr1[0] = (frag_dq_T[4] - cache_k_zero_point_smem[col_idx]) *
                           cache_k_scale_smem[col_idx];
          k_tile_ptr1[1] =
              (frag_dq_T[5] - cache_k_zero_point_smem[col_idx + 1]) *
              cache_k_scale_smem[col_idx + 1];
          k_tile_ptr1[8] =
              (frag_dq_T[6] - cache_k_zero_point_smem[col_idx + 8]) *
              cache_k_scale_smem[col_idx + 8];
          k_tile_ptr1[9] =
              (frag_dq_T[7] - cache_k_zero_point_smem[col_idx + 9]) *
              cache_k_scale_smem[col_idx + 9];
          k_tile_ptr1[16] =
              (frag_dq_T[12] - cache_k_zero_point_smem[col_idx + 16]) *
              cache_k_scale_smem[col_idx + 16];
          k_tile_ptr1[17] =
              (frag_dq_T[13] - cache_k_zero_point_smem[col_idx + 17]) *
              cache_k_scale_smem[col_idx + 17];
          k_tile_ptr1[24] =
              (frag_dq_T[14] - cache_k_zero_point_smem[col_idx + 24]) *
              cache_k_scale_smem[col_idx + 24];
          k_tile_ptr1[25] =
              (frag_dq_T[15] - cache_k_zero_point_smem[col_idx + 25]) *
              cache_k_scale_smem[col_idx + 25];
        }
        col_idx += 32;
      }
      k_smem_offset_r = k_smem.advance_offset_by_column<2, num_vecs_per_head_k>(
          k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head_k>(
            k_smem_offset_r) -
        4;
  }

  // ================v================
  smem_t v_smem(smem + BLOCK_SIZE * HEAD_DIM * sizeof(CacheT) / 2);
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + tid / 2, tid % 2);  // 4 * 8 per warp

  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t v_read_idx = (wid * 16 + tid / 2) * BLOCK_SIZE_HALF +
                        tid % 2 * num_elems_per_128b<CacheT>();
  // load v_smem 128 rows 64 rows
  for (int fy = 0; fy < 2;
       fy++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 2 iter
    for (int fz = 0; fz < 1; fz++) {  // 2 * 128b = 64 * int4 once, need 1 iter
      v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
          v_smem_offset_w, cur_cache_v + v_read_idx, end_idx > 0);
      v_smem_offset_w =
          v_smem.advance_offset_by_column<2, num_vecs_per_blocksize>(
              v_smem_offset_w, fz);
      v_read_idx += 2 * num_elems_per_128b<CacheT>();
    }
    v_smem_offset_w =
        v_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_blocksize>(
            v_smem_offset_w) -
        2;
    v_read_idx +=
        16 * NUM_WARPS * BLOCK_SIZE_HALF - 2 * num_elems_per_128b<CacheT>();
  }

  commit_group();
  wait_group<0>();
  __syncthreads();

  // deal v_smem 128 rows 64 cols
  for (int fy = 0; fy < 2;
       fy++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 2 iter
    uint32_t dim_idx = fy * NUM_WARPS * 16 + wid * 16 + tid / 4;
    for (int fz = 0; fz < 1; fz++) {  // 2 * 128b = 64 * int4 once, need 1 iter
      uint32_t kv_idx = fz * 64 + tid % 4 * 2;
      v_smem.ldmatrix_m8n8x4(v_smem_offset_r, v_frag);
      // layout
      for (int i = 0; i < 2; i++) {
        T *v_tile_ptr0 = v_write_ptr + kv_idx * kv_t_stride +
                         kv_head_idx * HEAD_DIM + dim_idx;
        T *v_tile_ptr1 = v_tile_ptr0 + 8;

        convert_int4(frag_dq_T, v_frag[2 * i]);
        convert_int4(frag_dq_T + 8, v_frag[2 * i + 1]);
        if (kv_idx < end_idx) {
          v_tile_ptr0[0] = (frag_dq_T[0] - cache_v_zero_point_smem[dim_idx]) *
                           cache_v_scale_smem[dim_idx];
          v_tile_ptr1[0] =
              (frag_dq_T[4] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 1 < end_idx) {
          v_tile_ptr0[kv_t_stride] =
              (frag_dq_T[1] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[kv_t_stride] =
              (frag_dq_T[5] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 8 < end_idx) {
          v_tile_ptr0[8 * kv_t_stride] =
              (frag_dq_T[2] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[8 * kv_t_stride] =
              (frag_dq_T[6] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 9 < end_idx) {
          v_tile_ptr0[9 * kv_t_stride] =
              (frag_dq_T[3] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[9 * kv_t_stride] =
              (frag_dq_T[7] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 16 < end_idx) {
          v_tile_ptr0[16 * kv_t_stride] =
              (frag_dq_T[8] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[16 * kv_t_stride] =
              (frag_dq_T[12] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 17 < end_idx) {
          v_tile_ptr0[17 * kv_t_stride] =
              (frag_dq_T[9] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[17 * kv_t_stride] =
              (frag_dq_T[13] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 24 < end_idx) {
          v_tile_ptr0[24 * kv_t_stride] =
              (frag_dq_T[10] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[24 * kv_t_stride] =
              (frag_dq_T[14] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        if (kv_idx + 25 < end_idx) {
          v_tile_ptr0[25 * kv_t_stride] =
              (frag_dq_T[11] - cache_v_zero_point_smem[dim_idx]) *
              cache_v_scale_smem[dim_idx];
          v_tile_ptr1[25 * kv_t_stride] =
              (frag_dq_T[15] - cache_v_zero_point_smem[dim_idx + 8]) *
              cache_v_scale_smem[dim_idx + 8];
        }
        kv_idx += 32;
      }
      v_smem_offset_r =
          v_smem.advance_offset_by_column<2, num_vecs_per_blocksize>(
              v_smem_offset_r, fz);
    }
    v_smem_offset_r =
        v_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_blocksize>(
            v_smem_offset_r) -
        2;
  }
}

template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
void AppendCacheKV(const paddle::Tensor &cache_k,
                   const paddle::Tensor &cache_v,
                   const paddle::Tensor &cache_k_dequant_scales,
                   const paddle::Tensor &cache_v_dequant_scales,
                   const paddle::Tensor &cache_k_zp,
                   const paddle::Tensor &cache_v_zp,
                   const paddle::Tensor &seq_lens_this_time,
                   const paddle::Tensor &seq_lens_decoder,
                   const paddle::Tensor &cu_seqlens_k,
                   const paddle::Tensor &block_tables,
                   const paddle::Tensor &cache_batch_ids,
                   const paddle::Tensor &cache_tile_ids_per_batch,
                   const paddle::Tensor &cache_num_blocks_x,
                   const int max_blocks_per_seq,
                   const int kv_num_heads,
                   const std::string &cache_quant_type,
                   paddle::Tensor *k_out,
                   paddle::Tensor *v_out,
                   const cudaStream_t &stream) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  constexpr int NUM_WARPS = 4;
  int block_num = cache_num_blocks_x.data<int>()[0];
  dim3 grids(block_num, 1, kv_num_heads);
  dim3 blocks(32, NUM_WARPS);
  if (cache_quant_type == "none") {
    const uint32_t smem_size = BLOCK_SIZE * HEAD_DIM * sizeof(T) * 2;
    auto kernel_func =
        append_cache_kv_c16<NV_TYPE, NV_TYPE, HEAD_DIM, BLOCK_SIZE, NUM_WARPS>;

    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    launchWithPdlWhenEnabled(
        kernel_func,
        grids,
        blocks,
        smem_size,
        stream,
        reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
        reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
        reinterpret_cast<NV_TYPE *>(k_out->data<T>()),
        reinterpret_cast<NV_TYPE *>(v_out->data<T>()),
        seq_lens_this_time.data<int>(),
        seq_lens_decoder.data<int>(),
        cu_seqlens_k.data<int>(),
        block_tables.data<int>(),
        cache_batch_ids.data<int>(),
        cache_tile_ids_per_batch.data<int>(),
        max_blocks_per_seq,
        kv_num_heads);
  } else if (cache_quant_type == "cache_int8" ||
             cache_quant_type == "cache_fp8") {
    const uint32_t smem_size = BLOCK_SIZE * HEAD_DIM * sizeof(uint8_t) * 2;

    auto kernel_func = append_cache_kv_c8<NV_TYPE,
                                          uint8_t,
                                          HEAD_DIM,
                                          BLOCK_SIZE,
                                          NUM_WARPS,
                                          false>;
    if (cache_quant_type == "cache_fp8") {
      kernel_func = append_cache_kv_c8<NV_TYPE,
                                       uint8_t,
                                       HEAD_DIM,
                                       BLOCK_SIZE,
                                       NUM_WARPS,
                                       true>;
    }
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    launchWithPdlWhenEnabled(kernel_func,
                             grids,
                             blocks,
                             smem_size,
                             stream,
                             cache_k.data<uint8_t>(),
                             cache_v.data<uint8_t>(),
                             reinterpret_cast<NV_TYPE *>(k_out->data<T>()),
                             reinterpret_cast<NV_TYPE *>(v_out->data<T>()),
                             reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                 cache_k_dequant_scales.data<T>())),
                             reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                 cache_v_dequant_scales.data<T>())),
                             seq_lens_this_time.data<int>(),
                             seq_lens_decoder.data<int>(),
                             cu_seqlens_k.data<int>(),
                             block_tables.data<int>(),
                             cache_batch_ids.data<int>(),
                             cache_tile_ids_per_batch.data<int>(),
                             max_blocks_per_seq,
                             kv_num_heads);
  } else if (cache_quant_type == "cache_int4_zp") {
    const uint32_t smem_size =
        BLOCK_SIZE * HEAD_DIM * sizeof(uint8_t) + 4 * HEAD_DIM * sizeof(T);

    auto kernel_func =
        append_cache_kv_c4<NV_TYPE, uint8_t, HEAD_DIM, BLOCK_SIZE, NUM_WARPS>;

    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    launchWithPdlWhenEnabled(
        kernel_func,
        grids,
        blocks,
        smem_size,
        stream,
        cache_k.data<uint8_t>(),
        cache_v.data<uint8_t>(),
        reinterpret_cast<NV_TYPE *>(k_out->data<T>()),
        reinterpret_cast<NV_TYPE *>(v_out->data<T>()),
        reinterpret_cast<NV_TYPE *>(
            const_cast<T *>(cache_k_dequant_scales.data<T>())),
        reinterpret_cast<NV_TYPE *>(
            const_cast<T *>(cache_v_dequant_scales.data<T>())),
        reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_zp.data<T>())),
        reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_zp.data<T>())),
        seq_lens_this_time.data<int>(),
        seq_lens_decoder.data<int>(),
        cu_seqlens_k.data<int>(),
        block_tables.data<int>(),
        cache_batch_ids.data<int>(),
        cache_tile_ids_per_batch.data<int>(),
        max_blocks_per_seq,
        kv_num_heads);
  } else {
    PADDLE_THROW("%s mode isn't implemented yet", cache_quant_type.c_str());
  }
}

std::vector<paddle::Tensor> GQARopeWriteCacheKernel(
    const paddle::Tensor &qkv,
    const paddle::Tensor &key_cache,
    const paddle::Tensor &value_cache,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &cu_seqlens_k,
    const paddle::Tensor &rotary_embs,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &block_tables,
    const paddle::Tensor &kv_batch_ids,
    const paddle::Tensor &kv_tile_ids,
    const paddle::Tensor &kv_num_blocks,
    const paddle::Tensor &cache_batch_ids,
    const paddle::Tensor &cache_tile_ids,
    const paddle::Tensor &cache_num_blocks,
    const paddle::optional<paddle::Tensor> &cache_k_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_zp,
    const paddle::optional<paddle::Tensor> &cache_v_zp,
    const paddle::optional<paddle::Tensor> &kv_signal_data,
    const int kv_token_num,
    const int max_seq_len,
    const std::string &cache_quant_type,
    const bool rope_3d) {
  typedef PDTraits<paddle::DataType::BFLOAT16> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int kv_num_blocks_data = kv_num_blocks.data<int>()[0];
  const auto &qkv_dims = qkv.dims();
  const auto &key_cache_dims = key_cache.dims();
  const int token_num = qkv_dims[0];
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int block_size = key_cache.dims()[2];
  const int batch_size = seq_lens_this_time.dims()[0];
  const int kv_num_heads = key_cache_dims[1];
  const int head_dim = cache_quant_type == "cache_int4_zp"
                           ? key_cache_dims[3] * 2
                           : key_cache_dims[3];
  const int num_heads =
      qkv_dims[qkv_dims.size() - 1] / head_dim - 2 * kv_num_heads;
  const float softmax_scale = 1.f / sqrt(head_dim);

  AppendAttnMetaData meta_data;
  meta_data.token_nums = token_num;
  meta_data.kv_num_heads = kv_num_heads;
  meta_data.head_dims = head_dim;
  meta_data.q_num_heads = num_heads;
  meta_data.max_blocks_per_seq = max_blocks_per_seq;
  meta_data.block_size = block_size;
  meta_data.batch_size = seq_lens_this_time.dims()[0];

  phi::GPUContext *dev_ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(qkv.place()));

  auto stream = qkv.stream();
  paddle::Tensor qkv_out = GetEmptyTensor(qkv.dims(), qkv.dtype(), qkv.place());
  paddle::Tensor q = GetEmptyTensor(
      {token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor k = GetEmptyTensor(
      {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor v = GetEmptyTensor(
      {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());

  // rope
  gqa_rotary_qk_split_variable<data_t>(
      qkv_out.data<data_t>(),
      q.data<data_t>(),
      k.data<data_t>(),
      v.data<data_t>(),
      qkv.data<data_t>(),
      rotary_embs.data<float>(),
      batch_id_per_token.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      cu_seqlens_q.data<int>(),
      cu_seqlens_k.data<int>(),
      token_num,
      num_heads,
      kv_num_heads,
      max_seq_len,
      rope_3d ? rotary_embs.dims()[3] : rotary_embs.dims()[2],
      head_dim,
      rope_3d,
      stream);

  if (token_num < kv_token_num) {
    AppendCacheKV<data_t, 128, 64>(key_cache,
                                   value_cache,
                                   cache_k_dequant_scales.get(),
                                   cache_v_dequant_scales.get(),
                                   cache_k_zp.get(),
                                   cache_v_zp.get(),
                                   seq_lens_this_time,
                                   seq_lens_decoder,
                                   cu_seqlens_k,
                                   block_tables,
                                   cache_batch_ids,
                                   cache_tile_ids,
                                   cache_num_blocks,
                                   max_blocks_per_seq,
                                   kv_num_heads,
                                   cache_quant_type,
                                   &k,
                                   &v,
                                   stream);
  }
  // write cache
  if (cache_quant_type == "none") {
    CascadeAppendWriteCacheKVQKV<data_t>(
        meta_data,
        qkv_out,
        block_tables,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        max_seq_len,
        stream,
        const_cast<paddle::Tensor *>(&key_cache),
        const_cast<paddle::Tensor *>(&value_cache));
  } else if (cache_quant_type == "cache_int8" ||
             cache_quant_type == "cache_fp8" ||
             cache_quant_type == "block_wise_fp8") {
    CascadeAppendWriteCacheKVC8QKV<data_t, 128, 64>(
        meta_data,
        *const_cast<paddle::Tensor *>(&key_cache),
        *const_cast<paddle::Tensor *>(&value_cache),
        qkv_out,
        cache_k_quant_scales.get(),
        cache_v_quant_scales.get(),
        seq_lens_this_time,
        seq_lens_decoder,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        kv_batch_ids,
        kv_tile_ids,
        kv_num_blocks_data,
        max_seq_len,
        false,  // is_scale_channel_wise
        cache_quant_type,
        stream,
        const_cast<paddle::Tensor *>(&key_cache),
        const_cast<paddle::Tensor *>(&value_cache));
  } else if (cache_quant_type == "cache_int4_zp") {
    CascadeAppendWriteCacheKVC4QKV<data_t, 128, 64>(
        meta_data,
        *const_cast<paddle::Tensor *>(&key_cache),
        *const_cast<paddle::Tensor *>(&value_cache),
        qkv_out,
        cache_k_quant_scales.get(),
        cache_v_quant_scales.get(),
        cache_k_zp.get(),
        cache_v_zp.get(),
        seq_lens_this_time,
        seq_lens_decoder,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        kv_batch_ids,
        kv_tile_ids,
        kv_num_blocks_data,
        max_seq_len,
        stream,
        const_cast<paddle::Tensor *>(&key_cache),
        const_cast<paddle::Tensor *>(&value_cache));
  } else {
    PD_THROW(
        "cache_quant_type_str should be one of [none, cache_int8, cache_fp8, "
        "cache_int4_zp]");
  }
  const char *fmt_write_cache_completed_signal_str =
      std::getenv("FLAGS_fmt_write_cache_completed_signal");
  const char *FLAGS_use_pd_disaggregation_per_chunk =
      std::getenv("FLAGS_use_pd_disaggregation_per_chunk");
  if (fmt_write_cache_completed_signal_str &&
      (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
       std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
    if (FLAGS_use_pd_disaggregation_per_chunk &&
        (std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "true") == 0 ||
         std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "1") == 0)) {
      cudaLaunchHostFunc(
          qkv.stream(),
          &(RemoteCacheKvIpc::
                save_cache_kv_complete_signal_layerwise_per_query),
          (void *)nullptr);
    } else {
      if (kv_signal_data) {
        cudaLaunchHostFunc(
            qkv.stream(),
            &RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise,
            (void *)(const_cast<int64_t *>(
                kv_signal_data.get().data<int64_t>())));
      }
    }
  }
  return {q, k, v, qkv_out};
}

PD_BUILD_STATIC_OP(gqa_rope_write_cache)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "cu_seqlens_q",
             "cu_seqlens_k",
             "rotary_embs",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "batch_id_per_token",
             "block_tables",
             "kv_batch_ids",
             "kv_tile_ids_per_batch",
             "kv_num_blocks",
             "cache_batch_ids",
             "cache_tile_ids_per_batch",
             "cache_num_blocks",
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("kv_signal_data")})
    .Outputs({"q", "k", "v", "qkv_out", "key_cache_out", "value_cache_out"})
    .SetInplaceMap({{"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .Attrs({"kv_token_num: int",
            "max_seq_len: int",
            "cache_quant_type: std::string"})
    .SetKernelFn(PD_KERNEL(GQARopeWriteCacheKernel));
