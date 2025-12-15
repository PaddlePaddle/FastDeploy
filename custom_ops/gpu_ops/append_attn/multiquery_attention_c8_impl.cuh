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
#pragma once
#include "multiquery_attention_c8_kernel.h"
#include "cu_tensor_map.cuh"

template <typename T, typename CacheT>
struct Append_params {
  T *__restrict__ qkv;
  CacheT *__restrict__ cache_k;
  CacheT *__restrict__ cache_v;
  T *__restrict__ cache_k_scale;
  T *__restrict__ cache_v_scale;
  int *__restrict__ seq_lens_q;
  int *__restrict__ seq_lens_kv;
  int *__restrict__ batch_ids;
  int *__restrict__ tile_ids_per_batch;
  int *__restrict__ cu_seqlens_q;
  int *__restrict__ block_table;
  int *__restrict__ mask_offset;
  bool *__restrict__ attn_mask;
  T *__restrict__ tmp_o;
  float *__restrict__ tmp_m;
  float *__restrict__ tmp_d;
  int max_model_len;
  int max_kv_len;
  int max_block_num_per_seq;
  float softmax_scale;
  float quant_max_bound;
  float quant_min_bound;
  int chunk_size;
  int num_blocks_x;
  int token_num_per_batch;
  int attn_mask_len;
  bool sliding_window;
  int q_num_heads;
  int kv_num_heads;
  int max_num_chunks;
  int max_tile_q;
  int batch_size;
};

template <typename T, typename CacheT>
void print_params(Append_params<T, CacheT> const params) {
  printf("max_model_len: %d\n", params.max_model_len);
  printf("max_kv_len: %d\n", params.max_kv_len);
  printf("max_block_num_per_seq: %d\n", params.max_block_num_per_seq);
  printf("softmax_scale: %f\n", params.softmax_scale);
  printf("quant_max_bound: %f\n", params.quant_max_bound);
  printf("quant_min_bound: %f\n", params.quant_min_bound);
  printf("chunk_size: %d\n", params.chunk_size);
  printf("num_blocks_x: %d\n", params.num_blocks_x);
  printf("token_num_per_batch: %d\n", params.token_num_per_batch);
  printf("attn_mask_len: %d\n", params.attn_mask_len);
  printf("sliding_window: %d\n", params.sliding_window);
  printf("q_num_heads: %d\n", params.q_num_heads);
  printf("kv_num_heads: %d\n", params.kv_num_heads);
  printf("max_num_chunks: %d\n", params.max_num_chunks);
  printf("max_tile_q: %d\n", params.max_tile_q);
  printf("batch_size: %d\n", params.batch_size);
  
}

// __launch_bounds__(
//     NUM_THREADS_PER_BLOCK, 1
//   )
template <typename T,
          typename CacheT,
          bool partition_kv,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y,
          typename OutT = T,
          bool ENABLE_PREFILL = true,
          bool is_scale_channel_wise = false,
          bool IsFP8 = false,
          bool IsDynamicC8 = false>
__global__ void multi_query_append_attention_c8_warp1_4_kernel(
    const __grid_constant__ Append_params<T, CacheT> params,
    const __grid_constant__ CUtensorMap key_tensor_map,
    const __grid_constant__ CUtensorMap value_tensor_map
  ) {
  // constexpr uint32_t num_x = 1;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  // __syncthreads();
  // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
  //   printf("kernel start!");
  // }
  // __syncthreads();
  // 内存分配
  extern __shared__ __align__(128) uint8_t smem[];
  smem_t qo_smem(smem);
  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
  v_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
            NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT));
  smem_t k_scale_smem;
  smem_t v_scale_smem;
  T *k_smem_scale_ptr = nullptr;
  T *v_smem_scale_ptr = nullptr;

  // TMA
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar[4];
  if(tid == 0 && wid == 0) {
    for (int i = 0; i < 4; ++i) {
      init(&(bar[i]), blockDim.x * blockDim.y);
      cde::fence_proxy_async_shared_cta();
    }
  }
  __syncthreads();

  // 循环参数
  // block:[batch, kv_num_head, max_num_chunks, max_tile_q]
  int max_block_per_head = params.max_tile_q * params.max_num_chunks;
  int max_block_per_batch = max_block_per_head * params.kv_num_heads;
  int total_block = params.batch_size * max_block_per_batch;
  const int num_block = gridDim.x;
  // __syncthreads();
  // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
  //   printf("num_frags_x: %d\n", num_frags_x);
  // }
  // __syncthreads();
  for (int lane_idx = blockIdx.x; lane_idx < total_block; lane_idx += num_block) {
    int batch_idx = lane_idx / max_block_per_batch;
    int lane_id_in_bsz = lane_idx % max_block_per_batch;
    int kv_head_idx = lane_id_in_bsz / max_block_per_head;
    int lane_id_in_head = lane_id_in_bsz % max_block_per_head;
    int chunk_idx = lane_id_in_head / params.max_tile_q;
    int tile_idx = lane_id_in_head % params.max_tile_q;
    int q_head_idx = kv_head_idx * GROUP_SIZE;
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("tile_idx: %d, q_head_idx: %d\n", tile_idx, q_head_idx);
    // }
    // __syncthreads();
    const uint32_t q_len = params.seq_lens_q[batch_idx];
    if (q_len <= 0) {
      continue;
    }
    const int *block_table_now = params.block_table + batch_idx * params.max_block_num_per_seq;

    T cache_k_scale_reg[IsDynamicC8 ? num_frags_z * 2 : (is_scale_channel_wise ? num_frags_y * 4 : 1)];
    T cache_v_scale_reg[IsDynamicC8 ? num_frags_z * 4 : (is_scale_channel_wise ? num_frags_y * 2 : 1)];
    if constexpr (!IsDynamicC8) {
      if constexpr (is_scale_channel_wise) {
        int scale_col_base = threadIdx.x % 4 * 2 + kv_head_idx * HEAD_DIM;
        const T *cache_k_scale_cur_head = params.cache_k_scale + scale_col_base;
        for (int i = 0; i < num_frags_y; ++i) {
          const int scale_idx = i * 16;
          cache_k_scale_reg[i * 4] = cache_k_scale_cur_head[scale_idx];
          cache_k_scale_reg[i * 4 + 1] = cache_k_scale_cur_head[scale_idx + 1];
          cache_k_scale_reg[i * 4 + 2] = cache_k_scale_cur_head[scale_idx + 8];
          cache_k_scale_reg[i * 4 + 3] = cache_k_scale_cur_head[scale_idx + 9];
        }
        scale_col_base = threadIdx.x / 4 + kv_head_idx * HEAD_DIM;
        const T *cache_v_scale_cur_head = params.cache_v_scale + scale_col_base;
        for (int i = 0; i < num_frags_y; ++i) {
          const int scale_idx = i * 16;
          cache_v_scale_reg[i * 2] = cache_v_scale_cur_head[scale_idx];
          cache_v_scale_reg[i * 2 + 1] = cache_v_scale_cur_head[scale_idx + 8];
        }
      } else {
        cache_k_scale_reg[0] = params.cache_k_scale[kv_head_idx];
        cache_v_scale_reg[0] = params.cache_v_scale[kv_head_idx];
      }
    }
    const uint32_t num_rows_per_block = num_frags_x * 16;
    const uint32_t q_end =
        min(q_len, div_up((tile_idx + 1) * num_rows_per_block, GROUP_SIZE));
    uint32_t kv_len = params.seq_lens_kv[batch_idx];
    if (ENABLE_PREFILL) {
      kv_len += q_len;
      if (kv_len <= 0) {
        continue;
      }
    } else {
      if (kv_len <= 0) {
        continue;
      }
      kv_len += q_len;
    }
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("q_len: %d, kv_len: %d, q_end: %d\n", q_len, kv_len, q_end);
    // }
    // __syncthreads();
    const uint32_t num_chunks_this_seq = div_up(kv_len, params.chunk_size);
    if (chunk_idx >= num_chunks_this_seq) {
      continue;
    }

    // 相关const变量
    barrier::arrival_token tokens[4];
    constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
    constexpr uint32_t num_vecs_per_head_k =
        HEAD_DIM / num_elems_per_128b<CacheT>();
    constexpr uint32_t num_vecs_per_blocksize =
        BLOCK_SIZE / num_elems_per_128b<CacheT>();
    constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
    constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;
    
    const uint32_t q_n_stride = params.q_num_heads * HEAD_DIM;
    const uint32_t q_ori_n_stride = (params.q_num_heads + params.kv_num_heads * 2) * HEAD_DIM;
    const uint32_t kv_n_stride = params.kv_num_heads * BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_b_stride = HEAD_DIM;
    const uint32_t kv_d_stride = BLOCK_SIZE;

    float s_frag[num_frags_x][num_frags_z][8];
    float o_frag[num_frags_x][num_frags_y][8];
    float m_frag[num_frags_x][2];
    float d_frag[num_frags_x][2];
    

    T *o_base_ptr_T = nullptr;

    const uint32_t chunk_start = partition_kv ? chunk_idx * params.chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min(kv_len, chunk_start + params.chunk_size) : kv_len;
    const uint32_t chunk_len = chunk_end - chunk_start;

    
    init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

    
    const uint32_t q_start_seq_id = params.cu_seqlens_q[batch_idx];
    const uint32_t q_base_seq_id_this_block = tile_idx * num_frags_x * 16;
    const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                              q_head_idx * HEAD_DIM +
                              tid % 8 * num_elems_per_128b<T>();
    const uint32_t o_offset = q_start_seq_id * q_n_stride +
                              q_head_idx * HEAD_DIM +
                              tid % 8 * num_elems_per_128b<T>();
    T *q_base_ptr = params.qkv + q_offset;

    if (ENABLE_PREFILL) {
      o_base_ptr_T = params.tmp_o + batch_idx * params.max_num_chunks * q_n_stride +
                    chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                    tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T = params.tmp_o + q_start_seq_id * params.max_num_chunks * q_n_stride +
                    chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                    tid % 8 * num_elems_per_128b<T>();
    }
    const int *mask_offset_this_seq =
        params.mask_offset ? params.mask_offset + q_start_seq_id * 2 : nullptr;
    
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("load q start!\n");
    // }
    // __syncthreads();
    uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
        tid % 16, tid / 16);  // 16 * 16
    load_q_global_smem_multi_warps<GROUP_SIZE,
                                  num_frags_x,
                                  num_frags_y,
                                  HEAD_DIM,
                                  T>(q_base_ptr,
                                      &qo_smem,
                                      q_base_seq_id_this_block,
                                      q_end,
                                      q_ori_n_stride,
                                      HEAD_DIM);
    commit_group();
    wait_group<0>();
    __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("load q end!\n");
    // }
    // __syncthreads();

    q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
        &qo_smem, params.softmax_scale);

    
    if constexpr (IsDynamicC8) {
      k_smem_scale_ptr = reinterpret_cast<T *>(
          smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
          NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) * 2);
      v_smem_scale_ptr = k_smem_scale_ptr + NUM_WARP_KV * num_frags_z * 16;
      k_scale_smem.base = reinterpret_cast<b128_t *>(k_smem_scale_ptr);
      v_scale_smem.base = reinterpret_cast<b128_t *>(v_smem_scale_ptr);
    }

    const uint32_t num_iterations = div_up(
        CAUSAL
            ? (min(chunk_len,
                  sub_if_greater_or_zero(
                      kv_len - q_len +
                          div_up((tile_idx + 1) * num_rows_per_block, GROUP_SIZE),
                      chunk_start)))
            : chunk_len,
        NUM_WARP_KV * num_frags_z * 16);
    const uint32_t mask_check_iteration =
        (CAUSAL? (min(chunk_len,
                  sub_if_greater_or_zero(
                      kv_len - q_len +
                          tile_idx * num_rows_per_block / GROUP_SIZE,
                      chunk_start)))
        : params.mask_offset ? 0
                      : chunk_len) /
        (NUM_WARP_KV * num_frags_z * 16);

    uint32_t k_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
            wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

    uint32_t v_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
            (wid / 2) * num_frags_y * 16 + 8 * (tid / 16) + tid % 8,
            (wid % 2) * num_frags_z + (tid % 16) / 8);

    // uint32_t k_smem_offset_w =
    //     smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
    //         wid * 4 + tid / 8, tid % 8);
    // uint32_t v_smem_offset_w =
    //     smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
    //         wid * 8 + tid / 4, tid % 4);

    uint32_t kv_idx_base = chunk_start;
    // const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
    //                                 (wid * 4 + tid / 8) * kv_b_stride +
    //                                 tid % 8 * num_elems_per_128b<CacheT>();
    // const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
    //                                 (wid * 8 + tid / 4) * kv_d_stride +
    //                                 tid % 4 * num_elems_per_128b<CacheT>();

    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("load kv start!\n");
    // }
    // __syncthreads();
    // load BLOCK_SIZE * HEAD_DIM each time

    // produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
    //                       NUM_WARPS,
    //                       BLOCK_SIZE,
    //                       num_frags_y,
    //                       num_frags_z,
    //                       NUM_WARP_Q>(k_smem,
    //                                   &k_smem_offset_w,
    //                                   params.cache_k,
    //                                   block_table_now,
    //                                   kv_head_idx,
    //                                   kv_n_stride,
    //                                   kv_h_stride,
    //                                   kv_b_stride,
    //                                   kv_idx_base,
    //                                   chunk_end,
    //                                   const_k_offset);
#pragma unroll 1
    for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load k kv_i:%d, block_table_id!\n", kv_i, (kv_idx_base + kv_i * 64) / BLOCK_SIZE);
      // }
      // __syncthreads();
      int block_id = __ldg(&block_table_now[(kv_idx_base + kv_i * 64) / BLOCK_SIZE]);
      if (block_id < 0) block_id = 0;
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load k kv_i:%d, block_id:%d!\n", kv_i, block_id);
      // }
      // __syncthreads();
      if (tid == 0 && wid == 0) {
        // 发起 TMA 四维异步拷贝操作
        cde::cp_async_bulk_tensor_4d_global_to_shared((void*)(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT))), &key_tensor_map, 0, 0, kv_head_idx, block_id, bar[kv_i]);
        // 设置同步等待点，指定需要等待的拷贝完成的字节数。
        tokens[kv_i] = cuda::device::barrier_arrive_tx(bar[kv_i], 1, NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT));
        // printf("t0 barrier_arrive_tx end\n");
      } else {
        // Other threads just arrive.
        tokens[kv_i] = bar[kv_i].arrive();
        // printf("t1 arrive end token:%d\n", token);
      }
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load k launch end!\n");
      // }
      // __syncthreads();
    }
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("load k0  end!\n");
    // }
    // __syncthreads();

    if constexpr (IsDynamicC8) {
      produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                              BLOCK_SIZE,
                                              num_frags_z,
                                              NUM_WARP_Q>(k_scale_smem,
                                                          block_table_now,
                                                          params.cache_k_scale,
                                                          kv_idx_base,
                                                          params.kv_num_heads,
                                                          kv_head_idx,
                                                          chunk_end);
      commit_group();
    }
    
    
    // produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
    //                       NUM_WARPS,
    //                       BLOCK_SIZE,
    //                       num_frags_y,
    //                       num_frags_z,
    //                       NUM_WARP_Q>(v_smem,
    //                                   &v_smem_offset_w,
    //                                   params.cache_v,
    //                                   block_table_now,
    //                                   kv_head_idx,
    //                                   kv_n_stride,
    //                                   kv_h_stride,
    //                                   kv_d_stride,
    //                                   kv_idx_base,
    //                                   chunk_end,
    //                                   const_v_offset);
#pragma unroll 1
    for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load v kv_i:%d, block_table_id!\n", kv_i, (kv_idx_base + kv_i * 64) / BLOCK_SIZE);
      // }
      // __syncthreads();
      int block_id = __ldg(&block_table_now[(kv_idx_base + kv_i * 64) / BLOCK_SIZE]);
      if (block_id < 0) block_id = 0;
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load v kv_i:%d, block_id:%d!\n", kv_i, block_id);
      // }
      // __syncthreads();
      if (tid == 0 && wid == 0) {
        // 发起 TMA 四维异步拷贝操作
        // printf("kv_i:%d, block_id:%d, kv_head_idx:%d, smem:%d\n", kv_i, block_id, kv_head_idx, static_cast<int32_t>(num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        //     NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT))));
        // printf("smem_ptr:%p\n", smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        //     NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT)));
        cde::cp_async_bulk_tensor_4d_global_to_shared(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
            NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT)), &value_tensor_map, 0, 0, kv_head_idx, block_id, bar[2 + kv_i]);
        // 设置同步等待点，指定需要等待的拷贝完成的字节数。
        // printf("bit:%d", NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT));
        tokens[2 + kv_i] = cuda::device::barrier_arrive_tx(bar[2 + kv_i], 1, NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT));
      } else {
        // Other threads just arrive.
        tokens[2 + kv_i] = bar[2 + kv_i].arrive();
      }
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("load v launch end!\n");
      // }
      // __syncthreads();
    }
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("load v0  end!\n");
    // }
    // __syncthreads();

    if constexpr (IsDynamicC8) {
      produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                              BLOCK_SIZE,
                                              num_frags_z,
                                              NUM_WARP_Q>(v_scale_smem,
                                                          block_table_now,
                                                          params.cache_v_scale,
                                                          kv_idx_base,
                                                          params.kv_num_heads,
                                                          kv_head_idx,
                                                          chunk_end);
      commit_group();
    }
    
#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      // wait_group<1>();
      
      if constexpr (IsDynamicC8) {
        wait_group<1>();
        __syncthreads();
        produce_k_dynamic_scale_smem2reg<BLOCK_SIZE, num_frags_z, NUM_WARP_Q, T>(
            k_smem_scale_ptr, cache_k_scale_reg);
      }

      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("compute qk start!\n");
      // }
      // __syncthreads();
      // s = qk
#pragma unroll 1
      for(uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
        bar[kv_i].wait(std::move(tokens[kv_i]));
      }
      compute_qk_c8<num_frags_x,
                    num_frags_y,
                    num_frags_z,
                    T,
                    CacheT,
                    is_scale_channel_wise,
                    IsFP8,
                    IsDynamicC8>(&qo_smem,
                                &q_smem_offset_r,
                                &k_smem,
                                &k_smem_offset_r,
                                cache_k_scale_reg,
                                s_frag);
      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("compute qk end!\n");
      // }
      // __syncthreads();
      // mask according to kv_idx and q_idx
      if (iter >= mask_check_iteration || params.sliding_window > 0) {
        mask_s<T,
              partition_kv,
              CAUSAL,
              GROUP_SIZE,
              NUM_WARPS,
              num_frags_x,
              num_frags_y,
              num_frags_z>(
            params.attn_mask ? params.attn_mask + batch_idx * params.attn_mask_len * params.attn_mask_len
                      : nullptr,
            q_base_seq_id_this_block,
            kv_idx_base + wid * num_frags_z * 16,
            q_len,
            kv_len,
            chunk_end,
            params.attn_mask_len,
            s_frag,
            mask_offset_this_seq,
            params.sliding_window);
      }

      // update m,d
      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
          s_frag, o_frag, m_frag, d_frag);
      __syncthreads();

      // const uint32_t ori_kv_idx_base = kv_idx_base;
      kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
      // produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
      //                       NUM_WARPS,
      //                       BLOCK_SIZE,
      //                       num_frags_y,
      //                       num_frags_z,
      //                       NUM_WARP_Q>(k_smem,
      //                                   &k_smem_offset_w,
      //                                   params.cache_k,
      //                                   block_table_now,
      //                                   kv_head_idx,
      //                                   kv_n_stride,
      //                                   kv_h_stride,
      //                                   kv_b_stride,
      //                                   kv_idx_base,
      //                                   chunk_end,
      //                                   const_k_offset);
#pragma unroll 1
      for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
        int block_id = __ldg(&block_table_now[(kv_idx_base + kv_i * 64) / BLOCK_SIZE]);
        if (block_id < 0) block_id = 0;
        if (tid == 0 && wid == 0) {
          // 发起 TMA 四维异步拷贝操作
          cde::cp_async_bulk_tensor_4d_global_to_shared(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT)), &key_tensor_map, 0, 0, kv_head_idx, block_id, bar[kv_i]);
          // 设置同步等待点，指定需要等待的拷贝完成的字节数。
          tokens[kv_i] = cuda::device::barrier_arrive_tx(bar[kv_i], 1, NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT));
        } else {
          // Other threads just arrive.
          tokens[kv_i] = bar[kv_i].arrive();
        }
      }

      if constexpr (IsDynamicC8) {
        produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                                BLOCK_SIZE,
                                                num_frags_z,
                                                NUM_WARP_Q>(k_scale_smem,
                                                            block_table_now,
                                                            params.cache_k_scale,
                                                            kv_idx_base,
                                                            params.kv_num_heads,
                                                            kv_head_idx,
                                                            chunk_end);
        commit_group();
      }
      // commit_group();
      // wait_group<1>();
      
      if constexpr (IsDynamicC8) {
        wait_group<1>();
        __syncthreads();
        produce_v_dynamic_scale_smem2reg<BLOCK_SIZE, num_frags_z, NUM_WARP_Q, T>(
            v_smem_scale_ptr, cache_v_scale_reg);
      }

      // __syncthreads();
      // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
      //   printf("compute sv start!\n");
      // }
      // __syncthreads();
#pragma unroll 1
      for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
        bar[2 + kv_i].wait(std::move(tokens[2 + kv_i]));
      }
      // compute sfm * v
      compute_sfm_v_c8_iter_sq_bvec<num_frags_x,
                                    num_frags_y,
                                    num_frags_z,
                                    BLOCK_SIZE,
                                    T,
                                    CacheT,
                                    is_scale_channel_wise,
                                    IsFP8,
                                    IsDynamicC8>(
          &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag, cache_v_scale_reg);
      __syncthreads();

      // produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
      //                       NUM_WARPS,
      //                       BLOCK_SIZE,
      //                       num_frags_y,
      //                       num_frags_z,
      //                       NUM_WARP_Q>(v_smem,
      //                                   &v_smem_offset_w,
      //                                   params.cache_v,
      //                                   block_table_now,
      //                                   kv_head_idx,
      //                                   kv_n_stride,
      //                                   kv_h_stride,
      //                                   kv_d_stride,
      //                                   kv_idx_base,
      //                                   chunk_end,
      //                                   const_v_offset);
#pragma unroll 1
      for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
        int block_id = __ldg(&block_table_now[(kv_idx_base + kv_i * 64) / BLOCK_SIZE]);
        if (block_id < 0) block_id = 0;
        if (tid == 0 && wid == 0) {
          // 发起 TMA 四维异步拷贝操作
          cde::cp_async_bulk_tensor_4d_global_to_shared(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
            NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) + kv_i * (NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT)), &value_tensor_map, 0, 0, kv_head_idx, block_id, bar[2 + kv_i]);
          // 设置同步等待点，指定需要等待的拷贝完成的字节数。
          tokens[2 + kv_i] = cuda::device::barrier_arrive_tx(bar[2 + kv_i], 1, NUM_WARP_KV * 16 * HEAD_DIM * sizeof(CacheT));
        } else {
          // Other threads just arrive.
          tokens[2 + kv_i] = bar[2 + kv_i].arrive();
        }
      }
      if constexpr (IsDynamicC8) {
        produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                                BLOCK_SIZE,
                                                num_frags_z,
                                                NUM_WARP_Q>(v_scale_smem,
                                                            block_table_now,
                                                            params.cache_v_scale,
                                                            kv_idx_base,
                                                            params.kv_num_heads,
                                                            kv_head_idx,
                                                            chunk_end);
        commit_group();
      }
      // commit_group();
    }
#pragma unroll 1
    for (uint32_t i = 0; i < NUM_WARP_KV; ++i) {
      bar[i].wait(std::move(tokens[i]));
    }
    if constexpr (IsDynamicC8) {
      wait_group<0>();
      __syncthreads();
    }
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("merge start!");
    // }
    // __syncthreads();
    merge_block_res_v2<num_frags_x, num_frags_y, T>(
        o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);

    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("merge end!");
    // }
    // __syncthreads();
    // write o
    // [num_frags_x, 16, num_frags_y, 16]
    write_o_reg_gmem_multi_warps<GROUP_SIZE,
                                 num_frags_x,
                                 num_frags_y,
                                 T>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        q_base_seq_id_this_block,
        q_head_idx,
        q_len,
        q_n_stride * params.max_num_chunks,
        HEAD_DIM);


    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("s_frag: [");
    //   for (int32_t fx = 0; fx < num_frags_x; ++fx) {
    //     for (int32_t fz = 0; fz < num_frags_z; ++fz) {
    //       for (int32_t fi = 0; fi < 8; ++fi) {
    //         printf("%.2f ", s_frag[fx][fz][fi]);
    //       }
    //     } 
    //   }
    //   printf("]\n");
    //   printf("o_frag: [");
    //   for (int32_t fx = 0; fx < num_frags_x; ++fx) {
    //     for (int32_t fy = 0; fy < num_frags_y; ++fy) {
    //       for (int32_t fi = 0; fi < 8; ++fi) {
    //         printf("%.2f ", o_frag[fx][fy][fi]);
    //       }
    //     } 
    //   }
    //   printf("]\n");
    //   printf("m_frags: [");
    //   for (int32_t fx = 0; fx < num_frags_x; ++fx) {
    //     for (int32_t fi = 0; fi < 2; ++fi) {
    //       printf("%.2f ", m_frag[fx][fi]);
    //     }
    //   }
    //   printf("]\n");
    //   printf("d_frags: [");
    //   for (int32_t fx = 0; fx < num_frags_x; ++fx) {
    //     for (int32_t fi = 0; fi < 8; ++fi) {
    //       printf("%.2f ", d_frag[fx][fi]);
    //     }
    //   }
    //   printf("]\n");
    // }
    // __syncthreads();
    if (wid == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_idx_now =
              q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
          const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
          const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
          if (qo_idx - q_start_seq_id < q_len) {
            uint32_t offset;
            if (ENABLE_PREFILL) {
              offset = (batch_idx * params.max_num_chunks + chunk_idx) * params.q_num_heads +
                      qo_head_idx;
            } else {
              offset =
                  (qo_idx * params.max_num_chunks + chunk_idx) * params.q_num_heads + qo_head_idx;
            }
            params.tmp_m[offset] = m_frag[fx][j];
            params.tmp_d[offset] = d_frag[fx][j];
          }
        }
      }
    }
    // __syncthreads();
    // if(blockIdx.x == 0 && tid == 0 && wid == 0) {
    //   printf("kernel end!\n");
    // }
    // __syncthreads();
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT,
          bool ENABLE_PREFILL,
          bool IsFP8,
          bool IsDynamicC8>
void MultiQueryAppendC8Attention(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &qkv,
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::Tensor &cache_k_scale,
    const paddle::Tensor &cache_v_scale,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::optional<paddle::Tensor> &sinks,
    const paddle::Tensor &seq_lens_q,
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &batch_id_per_token,
    const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_ids,
    const paddle::Tensor &tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t &stream,
    paddle::Tensor *out,
    const int sliding_window) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;


  constexpr uint32_t NUM_WARP_KV = NUM_WARPS_PER_BLOCK / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q); // BLOCK_SHAPE_Q=32
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto *allocator = paddle::GetAllocator(qkv.place());

  bool is_scale_channel_wise = false;
  if (cache_k_scale.dims()[0] == HEAD_DIM * kv_num_heads) {
    is_scale_channel_wise = true;
  }

  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 2;
  constexpr uint32_t smem_size_0 = num_frags_x * 16 * HEAD_DIM * sizeof(T) +
      NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2 +
      NUM_WARP_KV * num_frags_z * 16 * sizeof(T) * 2;
  constexpr uint32_t smem_size_1 = NUM_WARPS_PER_BLOCK * num_frags_x * num_frags_y * 32 * 8 * sizeof(float) + 
      NUM_WARPS_PER_BLOCK * num_frags_x * 2 * 32 * 8;
  constexpr uint32_t smem_size = smem_size_0 > smem_size_1 ? smem_size_0 : smem_size_1;

  auto split_kv_kernel =
      multi_query_append_attention_c8_warp1_4_kernel<NV_TYPE,
                                                      uint8_t,
                                                      true,
                                                      GROUP_SIZE,
                                                      CAUSAL,
                                                      NUM_WARPS_PER_BLOCK,
                                                      NUM_WARP_Q,
                                                      NUM_WARP_KV,
                                                      HEAD_DIM,
                                                      BLOCK_SIZE,
                                                      num_frags_x,
                                                      num_frags_z,
                                                      num_frags_y,
                                                      OUT_NV_TYPE,
                                                      ENABLE_PREFILL,
                                                      false,
                                                      IsFP8,
                                                      IsDynamicC8>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(split_kv_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          smem_size);
  }
  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);

  const int num_chunks = div_up(max_dec_len, chunk_size);
  uint32_t attn_mask_len;
  if (attn_mask) {
    attn_mask_len = attn_mask.get().shape()[1];
  } else {
    attn_mask_len = -1;
  }

  phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
  if (is_decoder) {
    tmp_workspace = allocator->Allocate(
        phi::SizeOf(qkv.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM));
    tmp_m = allocator->Allocate(
        phi::SizeOf(paddle::DataType::FLOAT32) *
        static_cast<size_t>(bsz * num_chunks * num_heads));
    tmp_d = allocator->Allocate(
        phi::SizeOf(paddle::DataType::FLOAT32) *
        static_cast<size_t>(bsz * num_chunks * num_heads));
  } else {
    tmp_workspace = allocator->Allocate(
        phi::SizeOf(qkv.dtype()) *
        static_cast<size_t>(token_num * num_chunks * num_heads * HEAD_DIM));
    tmp_m = allocator->Allocate(
        phi::SizeOf(paddle::DataType::FLOAT32) *
        static_cast<size_t>(token_num * num_chunks * num_heads));
    tmp_d = allocator->Allocate(
        phi::SizeOf(paddle::DataType::FLOAT32) *
        static_cast<size_t>(token_num * num_chunks * num_heads));
  }
  Append_params<NV_TYPE, uint8_t> params;
  memset(&params, 0, sizeof(Append_params<NV_TYPE, uint8_t>));

  params.qkv = reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>()));
  params.cache_k = const_cast<uint8_t *>(cache_k.data<uint8_t>());
  params.cache_v = const_cast<uint8_t *>(cache_v.data<uint8_t>());
  params.cache_k_scale = reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_scale.data<T>()));
  params.cache_v_scale = reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_scale.data<T>()));
  params.seq_lens_q = const_cast<int *>(seq_lens_q.data<int>());
  params.seq_lens_kv = const_cast<int *>(seq_lens_kv.data<int>());
  params.batch_ids = const_cast<int *>(batch_ids.data<int>());
  params.tile_ids_per_batch = const_cast<int *>(tile_ids_per_batch.data<int>());
  params.cu_seqlens_q = const_cast<int *>(cu_seqlens_q.data<int>());
  params.block_table = const_cast<int *>(block_table.data<int>());
  params.mask_offset = const_cast<int *>(meta_data.mask_offset);
  params.attn_mask = attn_mask ? const_cast<bool *>(attn_mask.get().data<bool>()) : nullptr;
  params.max_model_len = max_dec_len;
  params.max_kv_len = max_dec_len;
  params.max_block_num_per_seq = max_block_num_per_seq;
  params.softmax_scale = 1.f / sqrt(HEAD_DIM);
  params.quant_max_bound = quant_max_bound;
  params.quant_min_bound = quant_min_bound;
  params.chunk_size = chunk_size;
  params.tmp_o = reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr());
  params.tmp_m = static_cast<float *>(tmp_m->ptr());
  params.tmp_d = static_cast<float *>(tmp_d->ptr());
  params.token_num_per_batch = speculate_max_draft_token_num;
  params.attn_mask_len = attn_mask ? attn_mask_len = attn_mask.get().shape()[1] : -1;
  params.sliding_window = sliding_window;
  params.q_num_heads = num_heads;
  params.kv_num_heads = kv_num_heads;
  params.max_num_chunks = num_chunks;
  params.max_tile_q = div_up(GROUP_SIZE * speculate_max_draft_token_num, BLOCK_SHAPE_Q);
  params.batch_size = meta_data.batch_size;
  params.num_blocks_x = num_blocks_x_cpu;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_cout;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_cout, cudaDevAttrMultiProcessorCount, device));
  int total_num_block = params.batch_size * params.kv_num_heads * params.max_num_chunks * params.max_tile_q;
  dim3 grids(total_num_block);
  dim3 blocks(32, NUM_WARPS_PER_BLOCK);

  printf("grid: %d, smem_size: %d\n", grids.x, smem_size);
  print_params(params);
  printf("num_frags_x: %d, num_frags_y: %d, num_frags_z: %d, BLOCK_SHAPE_Q: %d\n", num_frags_x, num_frags_y, num_frags_z, BLOCK_SHAPE_Q);
  printf("NUM_WARP_Q: %d, NUM_WARP_KV: %d\n", NUM_WARP_Q, NUM_WARP_KV);
  cudaDeviceSynchronize();
  cudaCheckError();
  auto cache_k_dim = cache_k.dims();
  printf("cache_k_dims: [%d, %d, %d, %d]\n", cache_k_dim[0], cache_k_dim[1], cache_k_dim[2], cache_k_dim[3]);
  CUtensorMap key_tensor_map = makeTensorMapForKVCache<uint8_t>(cache_k.data<uint8_t>(), cache_k.dims()[0], params.kv_num_heads, BLOCK_SIZE, HEAD_DIM);
  
  printf("cache_k_map success\n");

  CUtensorMap value_tensor_map = makeTensorMapForKVCache<uint8_t>(cache_v.data<uint8_t>(), cache_v.dims()[0], params.kv_num_heads, HEAD_DIM, BLOCK_SIZE);
  printf("cache_v_map success\n");
  cudaDeviceSynchronize();
  cudaCheckError();
  
  
  launchWithPdlWhenEnabled(
      split_kv_kernel,
      grids,
      blocks,
      smem_size,
      stream,
      params,
      key_tensor_map,
      value_tensor_map);
  cudaDeviceSynchronize();
  cudaCheckError();
  // merge
  constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
  if (is_decoder) {
    constexpr int blockx = HEAD_DIM / vec_size;
    constexpr int blocky = (128 + blockx - 1) / blockx;
    dim3 grids_merge(bsz, num_heads);
    dim3 blocks_merge(blockx, blocky);
    auto *kernelFn = merge_multi_chunks_decoder_kernel<NV_TYPE,
                                                        vec_size,
                                                        blocky,
                                                        HEAD_DIM,
                                                        OUT_NV_TYPE,
                                                        ENABLE_PREFILL>;
    launchWithPdlWhenEnabled(
        kernelFn,
        grids_merge,
        blocks_merge,
        0,
        stream,
        reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
        static_cast<float *>(tmp_m->ptr()),
        static_cast<float *>(tmp_d->ptr()),
        seq_lens_q.data<int>(),
        seq_lens_kv.data<int>(),
        seq_lens_encoder.data<int>(),
        cu_seqlens_q.data<int>(),
        shift_bias ? reinterpret_cast<NV_TYPE *>(
                          const_cast<T *>(shift_bias.get().data<T>()))
                    : nullptr,
        smooth_weight ? reinterpret_cast<NV_TYPE *>(
                            const_cast<T *>(smooth_weight.get().data<T>()))
                      : nullptr,
        sinks ? reinterpret_cast<NV_TYPE *>(
                    const_cast<T *>(sinks.get().data<T>()))
              : nullptr,
        reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
        quant_max_bound,
        quant_min_bound,
        in_scale,
        max_seq_len,
        num_chunks,
        num_heads,
        chunk_size,
        HEAD_DIM);
  } else {
    constexpr int blockx = HEAD_DIM / vec_size;
    constexpr int blocky = (128 + blockx - 1) / blockx;
    dim3 grids_merge(min(sm_count * 4, token_num), num_heads);
    dim3 blocks_merge(blockx, blocky);
    launchWithPdlWhenEnabled(
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                      vec_size,
                                      blocky,
                                      HEAD_DIM,
                                      OUT_NV_TYPE,
                                      ENABLE_PREFILL>,
        grids_merge,
        blocks_merge,
        0,
        stream,
        reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
        static_cast<float *>(tmp_m->ptr()),
        static_cast<float *>(tmp_d->ptr()),
        seq_lens_q.data<int>(),
        seq_lens_kv.data<int>(),
        seq_lens_encoder.data<int>(),
        batch_id_per_token.data<int>(),
        cu_seqlens_q.data<int>(),
        shift_bias ? reinterpret_cast<NV_TYPE *>(
                          const_cast<T *>(shift_bias.get().data<T>()))
                    : nullptr,
        smooth_weight ? reinterpret_cast<NV_TYPE *>(
                            const_cast<T *>(smooth_weight.get().data<T>()))
                      : nullptr,
        sinks ? reinterpret_cast<NV_TYPE *>(
                    const_cast<T *>(sinks.get().data<T>()))
              : nullptr,
        reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
        quant_max_bound,
        quant_min_bound,
        in_scale,
        max_seq_len,
        num_chunks,
        num_heads,
        chunk_size,
        HEAD_DIM,
        token_num,
        speculate_max_draft_token_num);
  }
  // cudaDeviceSynchronize();
  // cudaCheckError();
}
