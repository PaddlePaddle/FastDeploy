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
#include "utils.cuh"
// #include "cu_tensor_map.cuh"
#include "attention_func.cuh"

template <typename T, typename CacheT>
void print_params(AttentionParams<T, CacheT> const params) {
  printf("max_model_len: %d\n", params.max_model_len);
  printf("max_kv_len: %d\n", params.max_kv_len);
  printf("max_blocks_per_seq: %d\n", params.max_blocks_per_seq);
  printf("softmax_scale: %f\n", params.softmax_scale);
  printf("quant_max_bound: %f\n", params.quant_max_bound);
  printf("quant_min_bound: %f\n", params.quant_min_bound);
  printf("max_tokens_per_batch: %d\n", params.max_tokens_per_batch);
  printf("attn_mask_len: %d\n", params.attn_mask_len);
  printf("sliding_window: %d\n", params.sliding_window);
  printf("q_num_heads: %d\n", params.q_num_heads);
  printf("kv_num_heads: %d\n", params.kv_num_heads);
  printf("max_num_chunks: %d\n", params.max_num_chunks);
  printf("max_tile_q: %d\n", params.max_tile_q);
  printf("batch_size: %d\n", params.batch_size);
}

template <typename T,
          typename CacheT,
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
          bool is_scale_channel_wise = false,
          bool IsFP8 = false,
          bool IsDynamicC8 = false>
__global__ void decode_unified_attention_c8_kernel(
    AttentionParams<T, CacheT> params) {
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;

  // Cache loop-invariant params fields into registers.
  // Pass-by-value (no __grid_constant__) allows the compiler to cache
  // struct fields, and explicit local variables guarantee no constant
  // cache pressure in the grid-stride loop.
  // Only cache frequently-used fields; rarely-used ones are accessed
  // via params.xxx to reduce register pressure (Scheme I-A.2).
  const auto qkv = params.qkv;
  const auto cache_k = params.cache_k;
  const auto cache_v = params.cache_v;
  const auto cache_k_scale = params.cache_k_scale;
  const auto cache_v_scale = params.cache_v_scale;
  const auto seq_lens_q = params.seq_lens_q;
  const auto seq_lens_kv = params.seq_lens_kv;
  const auto block_table = params.block_table;
  const auto cu_seqlens_q = params.cu_seqlens_q;
  const auto block_indices = params.block_indices;
  const auto mask_offset = params.mask_offset;
  const auto attn_mask = params.attn_mask;
  const auto tmp_o = params.tmp_o;
  const auto tmp_m = params.tmp_m;
  const auto tmp_d = params.tmp_d;
  const float softmax_scale = params.softmax_scale;
  const int q_num_heads = params.q_num_heads;
  const int kv_num_heads = params.kv_num_heads;

  extern __shared__ __align__(128) uint8_t smem[];
  smem_t qo_smem(smem);
  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
             NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT));
  smem_t k_scale_smem;
  smem_t v_scale_smem;
  T* k_smem_scale_ptr = nullptr;
  T* v_smem_scale_ptr = nullptr;

  int total_block = params.num_blocks_ptr[0];
  int chunk_size = params.chunk_size_ptr[0];

  for (int lane_idx = blockIdx.x; lane_idx < total_block;
       lane_idx += gridDim.x) {
    int4 indices = reinterpret_cast<const int4*>(block_indices)[lane_idx];
    int batch_idx = indices.x;
    int kv_head_idx = indices.y;
    int chunk_idx = indices.z;
    int tile_idx = indices.w;
    int q_head_idx = kv_head_idx * GROUP_SIZE;

    const uint32_t q_len = seq_lens_q[batch_idx];
    const int* block_table_now =
        block_table + batch_idx * params.max_blocks_per_seq;

    T cache_k_scale_reg[IsDynamicC8
                            ? num_frags_z * 2
                            : (is_scale_channel_wise ? num_frags_y * 4 : 1)];
    T cache_v_scale_reg[IsDynamicC8
                            ? num_frags_z * 4
                            : (is_scale_channel_wise ? num_frags_y * 2 : 1)];
    if constexpr (!IsDynamicC8) {
      if constexpr (is_scale_channel_wise) {
        int scale_col_base = threadIdx.x % 4 * 2 + kv_head_idx * HEAD_DIM;
        const T* cache_k_scale_cur_head = cache_k_scale + scale_col_base;
        for (int i = 0; i < num_frags_y; ++i) {
          const int scale_idx = i * 16;
          cache_k_scale_reg[i * 4] = cache_k_scale_cur_head[scale_idx];
          cache_k_scale_reg[i * 4 + 1] = cache_k_scale_cur_head[scale_idx + 1];
          cache_k_scale_reg[i * 4 + 2] = cache_k_scale_cur_head[scale_idx + 8];
          cache_k_scale_reg[i * 4 + 3] = cache_k_scale_cur_head[scale_idx + 9];
        }
        scale_col_base = threadIdx.x / 4 + kv_head_idx * HEAD_DIM;
        const T* cache_v_scale_cur_head = cache_v_scale + scale_col_base;
        for (int i = 0; i < num_frags_y; ++i) {
          const int scale_idx = i * 16;
          cache_v_scale_reg[i * 2] = cache_v_scale_cur_head[scale_idx];
          cache_v_scale_reg[i * 2 + 1] = cache_v_scale_cur_head[scale_idx + 8];
        }
      } else {
        cache_k_scale_reg[0] = cache_k_scale[kv_head_idx];
        cache_v_scale_reg[0] = cache_v_scale[kv_head_idx];
      }
    }
    constexpr uint32_t num_rows_per_block = num_frags_x * 16;
    const uint32_t q_end =
        min(q_len, div_up((tile_idx + 1) * num_rows_per_block, GROUP_SIZE));
    const uint32_t kv_len = seq_lens_kv[batch_idx] + q_len;
    const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);

    constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
    constexpr uint32_t num_vecs_per_head_k =
        HEAD_DIM / num_elems_per_128b<CacheT>();
    constexpr uint32_t num_vecs_per_blocksize =
        BLOCK_SIZE / num_elems_per_128b<CacheT>();
    constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
    constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;

    const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
    const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
    const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
    const uint32_t kv_b_stride = HEAD_DIM;
    const uint32_t kv_d_stride = BLOCK_SIZE;

    float s_frag[num_frags_x][num_frags_z][8];
    float o_frag[num_frags_x][num_frags_y][8];
    float m_frag[num_frags_x][2];
    float d_frag[num_frags_x][2];

    T* o_base_ptr_T = nullptr;

    const uint32_t chunk_start = chunk_idx * chunk_size;
    const uint32_t chunk_end = min(kv_len, chunk_start + chunk_size);
    const uint32_t chunk_len = chunk_end - chunk_start;

    init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

    const uint32_t q_start_seq_id = cu_seqlens_q[batch_idx];
    const uint32_t q_base_seq_id_this_block = tile_idx * num_frags_x * 16;
    const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                              q_head_idx * HEAD_DIM +
                              tid % 8 * num_elems_per_128b<T>();
    T* q_base_ptr = qkv + q_offset;

    o_base_ptr_T = tmp_o +
                   batch_idx * params.max_tokens_per_batch *
                       params.max_num_chunks * q_n_stride +
                   chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                   tid % 8 * num_elems_per_128b<T>();
    const int* mask_offset_this_seq =
        mask_offset ? mask_offset + q_start_seq_id * 2 : nullptr;
    const bool* attn_mask_this_seq =
        attn_mask ? attn_mask +
                        batch_idx * params.attn_mask_len * params.attn_mask_len
                  : nullptr;

    uint32_t q_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_head>(tid % 16, tid / 16);
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

    q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
        &qo_smem, softmax_scale);

    if constexpr (IsDynamicC8) {
      k_smem_scale_ptr = reinterpret_cast<T*>(
          smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
          NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT) * 2);
      v_smem_scale_ptr = k_smem_scale_ptr + NUM_WARP_KV * num_frags_z * 16;
      k_scale_smem.base = reinterpret_cast<b128_t*>(k_smem_scale_ptr);
      v_scale_smem.base = reinterpret_cast<b128_t*>(v_smem_scale_ptr);
    }

    const uint32_t num_iterations =
        div_up(CAUSAL ? (min(chunk_len,
                             sub_if_greater_or_zero(
                                 kv_len - q_len +
                                     div_up((tile_idx + 1) * num_rows_per_block,
                                            GROUP_SIZE),
                                 chunk_start)))
                      : chunk_len,
               NUM_WARP_KV * num_frags_z * 16);
    const uint32_t mask_check_iteration =
        (CAUSAL        ? (min(chunk_len,
                       sub_if_greater_or_zero(
                           kv_len - q_len +
                               tile_idx * num_rows_per_block / GROUP_SIZE,
                           chunk_start)))
         : mask_offset ? 0
                       : chunk_len) /
        (NUM_WARP_KV * num_frags_z * 16);

    uint32_t k_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
            wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

    uint32_t v_smem_offset_r =
        smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
            (wid / 2) * num_frags_y * 16 + 8 * (tid / 16) + tid % 8,
            (wid % 2) * num_frags_z + (tid % 16) / 8);

    uint32_t k_smem_offset_w =
        smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
            wid * 4 + tid / 8, tid % 8);
    uint32_t v_smem_offset_w =
        smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
            wid * 8 + tid / 4, tid % 4);

    uint32_t kv_idx_base = chunk_start;
    const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                    (wid * 4 + tid / 8) * kv_b_stride +
                                    tid % 8 * num_elems_per_128b<CacheT>();
    const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                    (wid * 8 + tid / 4) * kv_d_stride +
                                    tid % 4 * num_elems_per_128b<CacheT>();

    produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(k_smem,
                                       &k_smem_offset_w,
                                       cache_k,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_b_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_k_offset);

    if constexpr (IsDynamicC8) {
      produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                               BLOCK_SIZE,
                                               num_frags_z,
                                               NUM_WARP_Q>(k_scale_smem,
                                                           block_table_now,
                                                           cache_k_scale,
                                                           kv_idx_base,
                                                           kv_num_heads,
                                                           kv_head_idx,
                                                           chunk_end);
    }
    commit_group();

    produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(v_smem,
                                       &v_smem_offset_w,
                                       cache_v,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_d_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_v_offset);

    if constexpr (IsDynamicC8) {
      produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                               BLOCK_SIZE,
                                               num_frags_z,
                                               NUM_WARP_Q>(v_scale_smem,
                                                           block_table_now,
                                                           cache_v_scale,
                                                           kv_idx_base,
                                                           kv_num_heads,
                                                           kv_head_idx,
                                                           chunk_end);
    }
    commit_group();
#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      wait_group<1>();
      __syncthreads();

      if constexpr (IsDynamicC8) {
        produce_k_dynamic_scale_smem2reg<BLOCK_SIZE,
                                         num_frags_z,
                                         NUM_WARP_Q,
                                         T>(k_smem_scale_ptr,
                                            cache_k_scale_reg);
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

      if (iter >= mask_check_iteration || params.sliding_window > 0) {
        mask_s<T,
               CAUSAL,
               GROUP_SIZE,
               NUM_WARPS,
               num_frags_x,
               num_frags_y,
               num_frags_z>(attn_mask_this_seq,
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

      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
          s_frag, o_frag, m_frag, d_frag);
      __syncthreads();

      kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
      produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
                             NUM_WARPS,
                             BLOCK_SIZE,
                             num_frags_y,
                             num_frags_z,
                             NUM_WARP_Q>(k_smem,
                                         &k_smem_offset_w,
                                         cache_k,
                                         block_table_now,
                                         kv_head_idx,
                                         kv_n_stride,
                                         kv_h_stride,
                                         kv_b_stride,
                                         kv_idx_base,
                                         chunk_end,
                                         const_k_offset);

      if constexpr (IsDynamicC8) {
        produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                                 BLOCK_SIZE,
                                                 num_frags_z,
                                                 NUM_WARP_Q>(k_scale_smem,
                                                             block_table_now,
                                                             cache_k_scale,
                                                             kv_idx_base,
                                                             kv_num_heads,
                                                             kv_head_idx,
                                                             chunk_end);
      }
      commit_group();
      wait_group<1>();
      __syncthreads();

      if constexpr (IsDynamicC8) {
        produce_v_dynamic_scale_smem2reg<BLOCK_SIZE,
                                         num_frags_z,
                                         NUM_WARP_Q,
                                         T>(v_smem_scale_ptr,
                                            cache_v_scale_reg);
      }

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

      produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
                             NUM_WARPS,
                             BLOCK_SIZE,
                             num_frags_y,
                             num_frags_z,
                             NUM_WARP_Q>(v_smem,
                                         &v_smem_offset_w,
                                         cache_v,
                                         block_table_now,
                                         kv_head_idx,
                                         kv_n_stride,
                                         kv_h_stride,
                                         kv_d_stride,
                                         kv_idx_base,
                                         chunk_end,
                                         const_v_offset);

      if constexpr (IsDynamicC8) {
        produce_kv_dynamic_scale_gmem2smem_async<SharedMemFillMode::kFillZero,
                                                 BLOCK_SIZE,
                                                 num_frags_z,
                                                 NUM_WARP_Q>(v_scale_smem,
                                                             block_table_now,
                                                             cache_v_scale,
                                                             kv_idx_base,
                                                             kv_num_heads,
                                                             kv_head_idx,
                                                             chunk_end);
      }
      commit_group();
    }
    wait_group<0>();
    __syncthreads();
    const bool do_normalize = (num_chunks_this_seq <= 1);
    merge_block_res<num_frags_x, num_frags_y, T>(o_frag,
                                                 reinterpret_cast<float*>(smem),
                                                 m_frag,
                                                 d_frag,
                                                 wid,
                                                 tid,
                                                 do_normalize);

    write_o_reg_gmem_multi_warps<GROUP_SIZE, num_frags_x, num_frags_y, T>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        q_base_seq_id_this_block,
        q_head_idx,
        q_len,
        q_n_stride * params.max_num_chunks,
        HEAD_DIM);

    if (num_chunks_this_seq > 1) {
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
              offset = ((batch_idx * params.max_tokens_per_batch +
                         qo_idx_now / GROUP_SIZE) *
                            params.max_num_chunks +
                        chunk_idx) *
                           q_num_heads +
                       qo_head_idx;
              tmp_m[offset] = m_frag[fx][j];
              tmp_d[offset] = d_frag[fx][j];
            }
          }
        }
      }
    }
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t Q_TILE_SIZE,
          bool IsFP8,
          bool IsDynamicC8>
void DecodeUnifiedC8Attention(const AppendAttnMetaData& meta_data,
                              const paddle::Tensor& qkv,
                              const paddle::Tensor& cache_k,
                              const paddle::Tensor& cache_v,
                              const paddle::Tensor& tmp_workspace,
                              const paddle::Tensor& tmp_m,
                              const paddle::Tensor& tmp_d,
                              const paddle::optional<paddle::Tensor>& attn_mask,
                              const paddle::Tensor& cache_k_scale,
                              const paddle::Tensor& cache_v_scale,
                              const paddle::optional<paddle::Tensor>& sinks,
                              const paddle::Tensor& seq_lens_q,
                              const paddle::Tensor& seq_lens_kv,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& batch_id_per_token,
                              const paddle::Tensor& cu_seqlens_q,
                              const paddle::Tensor& block_table,
                              const paddle::Tensor& block_indices,
                              const paddle::Tensor& num_blocks,
                              const paddle::Tensor& chunk_size,
                              const int max_seq_len,
                              const int max_dec_len,
                              const float quant_max_bound,
                              const float quant_min_bound,
                              const int max_tokens_per_batch,
                              cudaStream_t& stream,
                              paddle::Tensor* out,
                              const int sliding_window) {
  using NV_TYPE = typename type_traits<T>::nv_type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_num;
  auto bsz = meta_data.batch_size;
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t NUM_WARP_Q = 1;
  constexpr uint32_t NUM_WARP_KV = NUM_WARPS_PER_BLOCK / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = Q_TILE_SIZE / (16 * NUM_WARP_Q);
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  bool is_scale_channel_wise = false;
  if (cache_k_scale.dims()[0] == HEAD_DIM * kv_num_heads) {
    is_scale_channel_wise = true;
  }

  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 2;
  constexpr uint32_t smem_size_0 =
      num_frags_x * 16 * HEAD_DIM * sizeof(T) +
      NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2 +
      NUM_WARP_KV * num_frags_z * 16 * sizeof(T) * 2;
  constexpr uint32_t smem_size_1 =
      NUM_WARPS_PER_BLOCK * num_frags_x * num_frags_y * 33 * 8 * sizeof(float) +
      NUM_WARPS_PER_BLOCK * num_frags_x * 2 * 33 * 8;
  constexpr uint32_t smem_size =
      smem_size_0 > smem_size_1 ? smem_size_0 : smem_size_1;

  auto split_kv_kernel = decode_unified_attention_c8_kernel<NV_TYPE,
                                                            uint8_t,
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
                                                            false,
                                                            IsFP8,
                                                            IsDynamicC8>;
  if (is_scale_channel_wise) {
    split_kv_kernel = decode_unified_attention_c8_kernel<NV_TYPE,
                                                         uint8_t,
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
                                                         true,
                                                         IsFP8,
                                                         IsDynamicC8>;
  }
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(split_kv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }
  const int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  const int max_num_chunks = div_up(max_seq_len, 512);
  uint32_t attn_mask_len;
  if (attn_mask) {
    attn_mask_len = attn_mask.get().shape()[1];
  } else {
    attn_mask_len = -1;
  }

  AttentionParams<NV_TYPE, uint8_t> params;
  memset(&params, 0, sizeof(AttentionParams<NV_TYPE, uint8_t>));

  params.qkv = reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>()));
  params.cache_k = const_cast<uint8_t*>(cache_k.data<uint8_t>());
  params.cache_v = const_cast<uint8_t*>(cache_v.data<uint8_t>());
  params.cache_k_scale =
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>()));
  params.cache_v_scale =
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>()));
  params.seq_lens_q = const_cast<int*>(seq_lens_q.data<int>());
  params.seq_lens_kv = const_cast<int*>(seq_lens_kv.data<int>());
  params.block_indices = const_cast<int*>(block_indices.data<int>());
  params.num_blocks_ptr = const_cast<int*>(num_blocks.data<int>());
  params.chunk_size_ptr = const_cast<int*>(chunk_size.data<int>());
  params.cu_seqlens_q = const_cast<int*>(cu_seqlens_q.data<int>());
  params.block_table = const_cast<int*>(block_table.data<int>());
  params.mask_offset = const_cast<int*>(meta_data.mask_offset);
  params.attn_mask =
      attn_mask ? const_cast<bool*>(attn_mask.get().data<bool>()) : nullptr;
  params.max_model_len = max_dec_len;
  params.max_kv_len = max_dec_len;
  params.max_blocks_per_seq = max_blocks_per_seq;
  params.softmax_scale = 1.f / sqrt(HEAD_DIM);
  params.quant_max_bound = quant_max_bound;
  params.quant_min_bound = quant_min_bound;
  params.tmp_o =
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(tmp_workspace.data<T>()));
  params.tmp_m = const_cast<float*>(tmp_m.data<float>());
  params.tmp_d = const_cast<float*>(tmp_d.data<float>());
  params.max_tokens_per_batch = max_tokens_per_batch;
  params.attn_mask_len =
      attn_mask ? attn_mask_len = attn_mask.get().shape()[1] : -1;
  params.sliding_window = sliding_window;
  params.q_num_heads = num_heads;
  params.kv_num_heads = kv_num_heads;
  params.max_num_chunks = max_num_chunks;
  params.batch_size = meta_data.batch_size;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_cout;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sm_cout, cudaDevAttrMultiProcessorCount, device));

  dim3 grids(sm_cout * 6);
  dim3 blocks(32, NUM_WARPS_PER_BLOCK);

  launchWithPdlWhenEnabled(
      split_kv_kernel, grids, blocks, smem_size, stream, params);

  constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
  constexpr int blockx = HEAD_DIM / vec_size;
  constexpr int blocky = (128 + blockx - 1) / blockx;
  dim3 grids_merge(min(sm_count * 4, token_num), num_heads);
  dim3 blocks_merge(blockx, blocky);
  launchWithPdlWhenEnabled(
      merge_chunks_kernel<NV_TYPE, vec_size, blocky, HEAD_DIM>,
      grids_merge,
      blocks_merge,
      0,
      stream,
      params.tmp_o,
      params.tmp_m,
      params.tmp_d,
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      seq_lens_encoder.data<int>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      (NV_TYPE*)nullptr,
      (NV_TYPE*)nullptr,
      sinks ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(sinks.get().data<T>()))
            : nullptr,
      chunk_size.data<int>(),
      reinterpret_cast<NV_TYPE*>(out->data<T>()),
      quant_max_bound,
      quant_min_bound,
      -1,
      max_seq_len,
      max_num_chunks,
      num_heads,
      HEAD_DIM,
      token_num,
      max_tokens_per_batch);
}
