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

#include "mma_tensor_op.cuh"
#include "utils.cuh"

template <typename T, uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void init_states(float (*o_frag)[num_frags_y][8],
                                            float (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if constexpr (std::is_same<T, half>::value) {
        m[fx][j] = -5e4f;
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        m[fx][j] = -3.0e+30f;
      }
      d[fx][j] = 1.f;
    }
  }
}

template <uint32_t BLOCK_SIZE>
__device__ __forceinline__ void load_block_table_per_chunk(
    const int32_t* block_table_chunk_start,
    int32_t* block_table_smem,
    uint32_t chunk_start,
    uint32_t chunk_end,
    uint32_t tid,
    uint32_t wid) {
  uint32_t len = chunk_end / BLOCK_SIZE - chunk_start / BLOCK_SIZE;
  for (uint32_t i = 0; i < div_up(len, 128); i++) {
    uint32_t offset = (wid * kWarpSize + tid) * i;
    if (offset <= len) {
      block_table_smem[offset] = block_table_chunk_start[offset];
    }
  }
}

// load q from global memory to shared memory
template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t HEAD_DIM,
          typename T>
__device__ __forceinline__ void load_q_global_smem_multi_warps(
    T* q_ptr_base,
    smem_t* q_smem,
    uint32_t q_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t q_smem_offset_w =  // [NUM_WARP_Q, num_frags_x, 16, head_dim]
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * 4 + tx / 8,
                                                     tx % 8);  // 4 * 64

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = q_idx_base + fx * 16 + tx_offset;
#pragma unroll
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;
    T* q_ptr = q_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
      q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
          q_smem_offset_w, q_ptr, n_offset < qo_upper_bound);
      q_smem_offset_w =
          q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
      q_ptr += 8 * num_elems_per_128b<T>();
    }
    q_smem_offset_w =
        q_smem->advance_offset_by_row<16, num_vecs_per_head>(q_smem_offset_w) -
        2 * num_frags_y;
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale_multi_warps(
    smem_t* q_smem,  // [num_frags_x * 16, num_frags_y * 16]
    const float sm_scale) {
  constexpr int vec_size = 16 / sizeof(T);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT tmp_vec;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 1024; ++i) {
    const int offset = i * 1024 + ty * 256 + tx * 8;
    Load<T, vec_size>(reinterpret_cast<T*>(q_smem->base) + offset, &tmp_vec);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp_vec[reg_id] *= sm_scale;
    }
    Store<T, vec_size>(tmp_vec, reinterpret_cast<T*>(q_smem->base) + offset);
  }
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_k_blockwise_c8(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_k,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_b_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_k_offset) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head =
      head_dim / num_elems_per_128b<CacheT>();  // 8
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;
#pragma unroll
  for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
    int block_id = __ldg(&block_table_now[kv_idx / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
#pragma unroll
    for (uint32_t i = 0; i < 2 * num_frags_z * 4 / num_warps;
         ++i) {  // m num_frags_z * 16 / (num_warps * 4)
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 8; ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, cache_k_now, true);
        *smem_offset = smem.advance_offset_by_column<8, num_vecs_per_head>(
            *smem_offset, j);
        cache_k_now += 8 * num_elems_per_128b<CacheT>();
      }
      kv_idx += num_warps * 4;
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
              *smem_offset) -
          num_frags_y;  // num_frags_y / 4 * 4
      cache_k_now += num_warps * 4 * kv_b_stride -
                     num_frags_y * num_elems_per_128b<CacheT>();
    }
  }
  *smem_offset -= NUM_WARP_KV * num_frags_z * 16 * num_vecs_per_head;
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_v_blockwise_c8(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_v,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_d_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_v_offset) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / num_elems_per_128b<CacheT>();  // 8
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + tx % 4 * num_elems_per_128b<CacheT>();

#pragma unroll
  for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
    int block_id = __ldg(&block_table_now[kv_idx / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;

#pragma unroll
    for (uint32_t i = 0; i < num_frags_y * 2 / num_warps;
         ++i) {  // m (num_frags_y * 16 / (num_warps * 8))
#pragma unroll
      for (uint32_t j = 0; j < 2 * num_frags_z / 4; ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, cache_v_now, true);
        *smem_offset = smem.advance_offset_by_column<4, num_vecs_per_blocksize>(
            *smem_offset, j);
        cache_v_now += 4 * num_elems_per_128b<CacheT>();
        kv_idx += 4 * num_elems_per_128b<CacheT>();
      }
      kv_idx -= 2 * num_frags_z * num_elems_per_128b<CacheT>();
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 8, num_vecs_per_blocksize>(
              *smem_offset) -
          2 * num_frags_z;  // num_frags_z / 4 * 4
      cache_v_now += num_warps * 8 * kv_d_stride -
                     2 * num_frags_z * num_elems_per_128b<CacheT>();
    }
    kv_idx += block_size;
  }
  *smem_offset -= NUM_WARP_KV / 2 * num_frags_y * 16 * num_vecs_per_blocksize;
}

template <SharedMemFillMode fill_mode,
          uint32_t block_size,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename T>
__device__ __forceinline__ void produce_kv_dynamic_scale_gmem2smem_async(
    smem_t kv_scale_smem,
    const int* block_table_now,
    const T* cache_kv_scale,
    const uint32_t kv_idx,
    const uint32_t kv_num_heads,
    const uint32_t kv_head_idx,
    const uint32_t chunk_end) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t tid = ty * 32 + tx;
  // 1 warp 32 tokens
  if (tid < block_size / 8 * 2) {
    const uint32_t kv_idx_now = kv_idx + block_size * tid / 8;
    int block_id = __ldg(&block_table_now[kv_idx_now / block_size]);
    if (block_id < 0) block_id = 0;
    const int kv_idx_this_thread = kv_idx + tid * 8;
    const T* cache_k_scale_now = cache_kv_scale +
                                 block_id * kv_num_heads * block_size +
                                 kv_head_idx * block_size + tid % 8 * 8;
    kv_scale_smem.load_128b_async<fill_mode>(
        tid, cache_k_scale_now, kv_idx_this_thread < chunk_end);
  }
}

template <uint32_t block_size,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename T>
__device__ __forceinline__ void produce_k_dynamic_scale_smem2reg(
    T* k_smem_scale, T* cache_k_reg) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  // 1 warp 32 tokens
  const uint32_t row_id = tx / 4;
  for (uint32_t fz = 0; fz < num_frags_z; fz++) {
    const uint32_t scale_idx = ty * 32 + fz * 16 + row_id;
    cache_k_reg[fz * 2] = k_smem_scale[scale_idx];
    cache_k_reg[fz * 2 + 1] = k_smem_scale[scale_idx + 8];
  }
}

template <uint32_t block_size,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename T>
__device__ __forceinline__ void produce_v_dynamic_scale_smem2reg(
    T* v_smem_scale, T* cache_v_reg) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  // 1 warp 32 tokens
  const uint32_t row_id = tx % 4 * 2;
  for (uint32_t fz = 0; fz < num_frags_z; fz++) {
    const uint32_t scale_idx = ty * 32 + fz * 16 + row_id;
    cache_v_reg[fz * 4] = v_smem_scale[scale_idx];
    cache_v_reg[fz * 4 + 1] = v_smem_scale[scale_idx + 1];
    cache_v_reg[fz * 4 + 2] = v_smem_scale[scale_idx + 8];
    cache_v_reg[fz * 4 + 3] = v_smem_scale[scale_idx + 9];
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T,
          typename CacheT,
          bool is_scale_channel_wise = false,
          bool IsFP8 = false,
          bool IsDynamicC8 = false>
__device__ __forceinline__ void compute_qk_c8(smem_t* q_smem,
                                              uint32_t* q_smem_offset_r,
                                              smem_t* k_smem,
                                              uint32_t* k_smem_offset_r,
                                              const T* cache_k_scale,
                                              float (*s_frag)[num_frags_z][8]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head_q = head_dim / num_elems_per_128b<T>();
  constexpr uint32_t num_vecs_per_head_k =
      head_dim / num_elems_per_128b<CacheT>();

  uint32_t a_frag[num_frags_x][2][4], b_frag[4], b_frag_dq[4];

#pragma unroll
  for (uint32_t ky = 0; ky < num_frags_y / 2; ++ky) {  // k
                                                       // load q
#pragma unroll
    for (uint32_t fy = 0; fy < 2; ++fy) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx][fy]);

        *q_smem_offset_r =
            q_smem->advance_offset_by_row<16, num_vecs_per_head_q>(
                *q_smem_offset_r);
      }
      *q_smem_offset_r =
          q_smem->advance_offset_by_column<2>(*q_smem_offset_r, ky * 2 + fy) -
          num_frags_x * 16 * num_vecs_per_head_q;
    }

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      // load
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, num_vecs_per_head_k>(
          *k_smem_offset_r);
#pragma unroll
      for (uint32_t fy = 0; fy < 2; ++fy) {
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_c8<T, IsFP8>(b_frag_dq_T, b_frag[fy * 2]);
        convert_c8<T, IsFP8>(b_frag_dq_T + 4, b_frag[fy * 2 + 1]);
        // scale zp
        if constexpr (!IsDynamicC8) {
          if constexpr (is_scale_channel_wise) {
            const int scale_col = (ky * 2 + fy) * 4;
            b_frag_dq_T[0] *= cache_k_scale[scale_col];
            b_frag_dq_T[1] *= cache_k_scale[scale_col + 1];
            b_frag_dq_T[2] *= cache_k_scale[scale_col + 2];
            b_frag_dq_T[3] *= cache_k_scale[scale_col + 3];
            b_frag_dq_T[4] *= cache_k_scale[scale_col];
            b_frag_dq_T[5] *= cache_k_scale[scale_col + 1];
            b_frag_dq_T[6] *= cache_k_scale[scale_col + 2];
            b_frag_dq_T[7] *= cache_k_scale[scale_col + 3];
          } else {
#pragma unroll
            for (uint32_t b_i = 0; b_i < 8; ++b_i) {
              b_frag_dq_T[b_i] *= cache_k_scale[0];
            }
          }
        } else {
#pragma unroll
          for (uint32_t b_i = 0; b_i < 8; ++b_i) {
            b_frag_dq_T[b_i] *= cache_k_scale[fz * 2 + b_i / 4];
          }
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          if (ky == 0 && fy == 0) {
            mma_sync_m16n16k16_row_col_f16f16f32<T, MMAMode::kInit>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          } else {
            mma_sync_m16n16k16_row_col_f16f16f32<T>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          }
        }
      }
    }
    *k_smem_offset_r = k_smem->advance_offset_by_column<2, num_vecs_per_head_k>(
                           *k_smem_offset_r, ky) -
                       num_frags_z * 16 * num_vecs_per_head_k;
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y / 2 * 2;
}

template <typename T,
          bool partition_kv,
          bool causal,
          uint32_t group_size,
          uint32_t num_warps,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z>
__device__ __forceinline__ void mask_s(const bool* attn_mask,
                                       const uint32_t qo_idx_base,
                                       const uint32_t kv_idx_base,
                                       const uint32_t qo_len,
                                       const uint32_t kv_len,
                                       const uint32_t chunk_end,
                                       const uint32_t attn_mask_len,
                                       float (*s_frag)[num_frags_z][8],
                                       const int* mask_offset = nullptr,
                                       const int sliding_window = 0) {
  const uint32_t tx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = (qo_idx_base + fx * 16 + tx / 4 +
                                8 * ((reg_id % 4) / 2)) /
                               group_size,
                       kv_idx = kv_idx_base + fz * 16 + 2 * (tx % 4) +
                                8 * (reg_id / 4) + reg_id % 2;
        bool out_of_boundary;
        if (mask_offset) {
          out_of_boundary = q_idx < qo_len
                                ? (kv_idx >= mask_offset[q_idx * 2 + 1] ||
                                   kv_idx < mask_offset[q_idx * 2])
                                : true;
        } else if (sliding_window > 0) {
          bool out_of_window = int(kv_idx) <= (int)kv_len + (int)q_idx -
                                                  (int)qo_len - sliding_window;
          out_of_boundary = (causal ? (kv_idx > kv_len + q_idx - qo_len ||
                                       out_of_window || (kv_idx >= chunk_end))
                                    : kv_idx >= chunk_end);
        } else {
          out_of_boundary = (causal ? (kv_idx > kv_len + q_idx - qo_len ||
                                       (kv_idx >= chunk_end))
                                    : kv_idx >= chunk_end);
          if (attn_mask != nullptr && kv_idx > kv_len - qo_len &&
              kv_idx < chunk_end && q_idx < attn_mask_len) {
            const int32_t mask_idx =
                q_idx * attn_mask_len + kv_idx - kv_len + qo_len;
            bool mask = attn_mask[mask_idx];
            out_of_boundary |= mask;
          }
        }

        if constexpr (std::is_same<T, half>::value) {
          s_frag[fx][fz][reg_id] =
              out_of_boundary ? -5e4f : s_frag[fx][fz][reg_id];
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
          s_frag[fx][fz][reg_id] =
              out_of_boundary ? -3.0e+30f : s_frag[fx][fz][reg_id];
        }
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void update_mdo_states(
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*m)[2],
    float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t j_id = j * 2;
      float m_prev = m[fx][j];
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        float* s_frag_tmp = s_frag[fx][fz] + j_id;
        float m_local = max(max(s_frag_tmp[0], s_frag_tmp[1]),
                            max(s_frag_tmp[4], s_frag_tmp[5]));
        m[fx][j] = max(m[fx][j], m_local);
      }
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x2, 32));
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x1, 32));
      float o_scale = expf(m_prev - m[fx][j]);
      d[fx][j] *= o_scale;
      float2 fp2_scale = make_float2(o_scale, o_scale);
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        // o_frag[fx][fy][j * 2 + 0] *= o_scale;
        // o_frag[fx][fy][j * 2 + 1] *= o_scale;
        // o_frag[fx][fy][j * 2 + 4] *= o_scale;
        // o_frag[fx][fy][j * 2 + 5] *= o_scale;

        float2* o_frag_ptr = reinterpret_cast<float2*>(o_frag[fx][fy] + j_id);
        // printf("fp2_len:%d, %d", sizeof(o_frag_ptr[0]), sizeof(fp2_scale));
        o_frag_ptr[0] = fast_float2_mul(o_frag_ptr[0], fp2_scale);
        o_frag_ptr[2] = fast_float2_mul(o_frag_ptr[2], fp2_scale);
      }
      float tmp_m = m[fx][j];
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        float* s_frag_ptr = s_frag[fx][fz] + j_id;
        s_frag_ptr[0] = __expf(s_frag_ptr[0] - tmp_m);
        s_frag_ptr[1] = __expf(s_frag_ptr[1] - tmp_m);
        s_frag_ptr[4] = __expf(s_frag_ptr[4] - tmp_m);
        s_frag_ptr[5] = __expf(s_frag_ptr[5] - tmp_m);
        // s_frag[fx][fz][j * 2 + 0] =
        //     __expf(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
        // s_frag[fx][fz][j * 2 + 1] =
        //     __expf(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
        // s_frag[fx][fz][j * 2 + 4] =
        //     __expf(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
        // s_frag[fx][fz][j * 2 + 5] =
        //     __expf(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
      }
    }
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t block_size,
          typename T,
          typename CacheT,
          bool is_scale_channel_wise = false,
          bool IsFP8 = false,
          bool IsDynamicC8 = false>
__device__ __forceinline__ void compute_sfm_v_c8_iter_sq_bvec(
    smem_t* v_smem,
    uint32_t* v_smem_offset_r,
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*d)[2],
    T* cache_v_scale) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / num_elems_per_128b<CacheT>();

  T s_frag_f16[num_frags_x][num_frags_z][8];
  uint32_t b_frag[4], b_frag_dq[4];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t kz = 0; kz < num_frags_z / 2; ++kz) {  // k
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      v_smem->ldmatrix_m8n8x4(*v_smem_offset_r, b_frag);
      *v_smem_offset_r =
          v_smem->advance_offset_by_row<16, num_vecs_per_blocksize>(
              *v_smem_offset_r);
#pragma unroll
      for (uint32_t fz = 0; fz < 2; ++fz) {
        // dequant b_frag -> b_frag_dq
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_c8<T, IsFP8>(b_frag_dq_T, b_frag[fz * 2]);
        convert_c8<T, IsFP8>(b_frag_dq_T + 4, b_frag[fz * 2 + 1]);
        // scale zp
        if constexpr (!IsDynamicC8) {
          if constexpr (is_scale_channel_wise) {
#pragma unroll
            for (uint32_t b_i = 0; b_i < 8; ++b_i) {
              b_frag_dq_T[b_i] *= cache_v_scale[b_i / 4 + fy * 2];
            }
          } else {
#pragma unroll
            for (uint32_t b_i = 0; b_i < 8; ++b_i) {
              b_frag_dq_T[b_i] *= cache_v_scale[0];
            }
          }
        } else {
          const int scale_col = (kz * 2 + fz) * 4;
          b_frag_dq_T[0] *= cache_v_scale[scale_col];
          b_frag_dq_T[1] *= cache_v_scale[scale_col + 1];
          b_frag_dq_T[2] *= cache_v_scale[scale_col + 2];
          b_frag_dq_T[3] *= cache_v_scale[scale_col + 3];
          b_frag_dq_T[4] *= cache_v_scale[scale_col];
          b_frag_dq_T[5] *= cache_v_scale[scale_col + 1];
          b_frag_dq_T[6] *= cache_v_scale[scale_col + 2];
          b_frag_dq_T[7] *= cache_v_scale[scale_col + 3];
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m: num_frags_x * 16
          mma_sync_m16n16k16_row_col_f16f16f32<T>(
              o_frag[fx][fy],
              (uint32_t*)(s_frag_f16[fx][kz * 2 + fz]),
              b_frag_dq);
        }
      }
    }
    *v_smem_offset_r -= num_frags_y * 16 * num_vecs_per_blocksize;
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void merge_block_res_v2(
    float (*o_frag)[num_frags_y][8],
    float* md_smem,
    float (*m)[2],
    float (*d)[2],
    const uint32_t wid,
    const uint32_t tid) {
  float2* smem_md = reinterpret_cast<float2*>(
      md_smem + num_frags_x * num_frags_y * 1024);  // 4 * 32 * 8
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      smem_md[((wid * num_frags_x + fx) * 2 + j) * 32 + tid] =
          make_float2(m[fx][j], d[fx][j]);
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      float2* md_smem_start =
          (float2*)(md_smem +
                    ((wid * num_frags_x + fx) * num_frags_y + fy) * 32 * 8 +
                    tid * 2);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        md_smem_start[i * 32] = ((float2*)(&o_frag[fx][fy][0]))[i];
      }
      // *(reinterpret_cast<float4*>(
      //     md_smem + (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 +
      //     tid) *
      //                   8)) =
      //                   *(reinterpret_cast<float4*>(&o_frag[fx][fy][0]));
      // *(reinterpret_cast<float4*>(
      //     md_smem +
      //     (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8 +
      //     4)) =
      // *(reinterpret_cast<float4*>(&o_frag[fx][fy][4]));
    }
  }
  __syncthreads();
  float o_scale[4][num_frags_x][2];

  // deal md/scale
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_new;
      float d_new = 1.f;
      if constexpr (std::is_same<T, half>::value) {
        m_new = -5e4f;
      } else {
        m_new = -3.0e+30f;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        // d_new = d_prev * expf(m_prev - m_new) + md.y * expf(md.x - m_new);
        d_new = fmaf(d_prev, expf(m_prev - m_new), md.y * expf(md.x - m_new));
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        o_scale[i][fx][j] = expf(md.x - m_new);
      }
      m[fx][j] = m_new;
      d[fx][j] = d_new;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // num_warps * 32 * 8 each time
      // AlignedVector<float2, 4> o_new_fp2;
      float2* o_new_fp2 = reinterpret_cast<float2*>(&o_frag[fx][fy][0]);
      // float2* o_new_fp2 = reinterpret_cast<float2*>(&o_new[0]);
#pragma
      for (uint32_t o_id = 0; o_id < 4; ++o_id) {
        o_new_fp2[o_id] = make_float2(0.f, 0.f);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        // AlignedVector<float, 8> oi;
        AlignedVector<float2, 4> oi_fp2;
        float2* md_smem_start =
            (float2*)(md_smem +
                      ((i * num_frags_x + fx) * num_frags_y + fy) * 32 * 8 +
                      tid * 2);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          oi_fp2[reg_id] = md_smem_start[reg_id * 32];
        }
#pragma unroll
        for (uint32_t reg_fp2_id = 0; reg_fp2_id < 4; ++reg_fp2_id) {
          float o_scale_fp2_tmp = o_scale[i][fx][reg_fp2_id % 2];
          o_new_fp2[reg_fp2_id] =
              fast_float2_fma(oi_fp2[reg_fp2_id],
                              make_float2(o_scale_fp2_tmp, o_scale_fp2_tmp),
                              o_new_fp2[reg_fp2_id]);
        }
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void merge_block_res_v21(
    float (*o_frag)[num_frags_y][8],
    float* md_smem,
    float (*m)[2],
    float (*d)[2],
    const uint32_t wid,
    const uint32_t tid) {
  float2* smem_md = reinterpret_cast<float2*>(
      md_smem + num_frags_x * num_frags_y * 1024);  // 4 * 32 * 8
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      smem_md[((wid * num_frags_x + fx) * 2 + j) * 32 + tid] =
          make_float2(m[fx][j], d[fx][j]);
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      *(reinterpret_cast<float4*>(
          md_smem + (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) *
                        8)) = *(reinterpret_cast<float4*>(&o_frag[fx][fy][0]));
      *(reinterpret_cast<float4*>(
          md_smem +
          (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8 + 4)) =
          *(reinterpret_cast<float4*>(&o_frag[fx][fy][4]));
    }
  }
  __syncthreads();
  float o_scale[4][num_frags_x][2];

  // deal md/scale
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_new;
      float d_new = 1.f;
      if constexpr (std::is_same<T, half>::value) {
        m_new = -5e4f;
      } else {
        m_new = -3.0e+30f;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        d_new = d_prev * __expf(m_prev - m_new) + md.y * __expf(md.x - m_new);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        o_scale[i][fx][j] = __expf(md.x - m_new);
      }
      m[fx][j] = m_new;
      d[fx][j] = d_new;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // num_warps * 32 * 8 each time
      AlignedVector<float, 8> o_new;
#pragma
      for (uint32_t o_id = 0; o_id < 4; ++o_id) {
        *(reinterpret_cast<float2*>(&o_new[o_id * 2])) = make_float2(0.f, 0.f);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        AlignedVector<float, 8> oi;
        Load<float, 8>(
            md_smem +
                (((i * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8,
            &oi);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_new[reg_id] += oi[reg_id] * o_scale[i][fx][(reg_id % 4) / 2];
        }
      }
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][0])) =
          *(reinterpret_cast<float4*>(&o_new[0]));
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][4])) =
          *(reinterpret_cast<float4*>(&o_new[4]));
    }
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          typename T,
          typename OutT>
__device__ __forceinline__ void write_o_reg_gmem_multi_warps(
    float (*o_frag)[num_frags_y][8],
    smem_t* o_smem,
    OutT* o_ptr_base,
    uint32_t o_idx_base,
    const uint32_t q_head_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr int VEC_SIZE = 16 / sizeof(T);
  // [num_warps * num_frags_x * 16, num_frags_y * 16]
  if (ty == 0) {
    // [num_frags_x * 16, num_frags_y * 16]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w =
            smem_t::get_permuted_offset<num_vecs_per_head>(fx * 16 + tx / 4,
                                                           fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] =
            o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[3];
      }
    }
  }
  __syncthreads();

  uint32_t o_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * 4 + tx / 8, tx % 8);

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = o_idx_base + fx * 16 + tx_offset;
#pragma unroll
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;

    OutT* o_ptr = o_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
      if (n_offset < qo_upper_bound) {
        o_smem->store_128b(o_smem_offset_w, o_ptr);
      }
      o_ptr += 8 * num_elems_per_128b<T>();
      o_smem_offset_w =
          o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
    }
    o_smem_offset_w =
        o_smem->advance_offset_by_row<16, num_vecs_per_head>(o_smem_offset_w) -
        2 * num_frags_y;
  }
}

template <size_t vec_size, typename T>
struct prefill_softmax_state_t {
  AlignedVector<T, vec_size> o;
  float m;
  float d;

  __device__ __forceinline__ void init() {
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2*)(&o) + i) = make_half2(0, 0);
      }
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162*)(&o) + i) = make_bfloat162(0, 0);
      }
    }
    d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = -3.38953e38f;
    }
  }

  __device__ __forceinline__ void merge(
      const AlignedVector<T, vec_size>& other_o, float other_m, float other_d) {
    float m_prev = m, d_prev = d;
    m = m_prev > other_m ? m_prev : other_m;
    const float scale1 = __expf(m_prev - m), scale2 = __expf(other_m - m);
    const T scale1_T = static_cast<T>(scale1),
            scale2_T = static_cast<T>(scale2);
    d = d_prev * scale1 + other_d * scale2;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1_T + other_o[i] * scale2_T;
    }
  }

  __device__ __forceinline__ void normalize() {
    const T d_t = static_cast<T>(d);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d_t;
    }
  }

  __device__ __forceinline__ void normalize(float current_sink) {
    const T d_t = static_cast<T>(d + __expf(current_sink - m));
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d_t;
    }
  }
};

template <typename T,
          int vec_size,
          uint32_t bdy,
          uint32_t HEAD_DIM,
          typename OutT = T>
__global__ void merge_multi_chunks_v2_kernel(
    const T* __restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float* __restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float* __restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int* __restrict__ seq_lens_q,
    const int* __restrict__ seq_lens_kv,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ batch_id_per_token,
    const int* __restrict__ cu_seqlens_q,
    const T* __restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T* __restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
    const T* __restrict__ sinks,          // [q_num_heads]
    const int* __restrict__ chunk_size_ptr,
    OutT* __restrict__ out,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int head_dim,
    const int token_num,
    const int max_tokens_per_batch = 5) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int qid = blockIdx.x; qid < token_num; qid += gridDim.x) {
    const uint32_t bid = batch_id_per_token[qid];
    if (bid == -1) {
      continue;
    }
    const uint32_t local_seq_id = qid - cu_seqlens_q[bid];
    const int seq_len_q = seq_lens_q[bid];
    if (seq_len_q == 0) continue;
    int seq_len_kv = seq_lens_kv[bid];
    if (seq_len_kv == 0) continue;
    seq_len_kv += seq_len_q;
    const int num_chunks_this_seq = div_up(seq_len_kv, *chunk_size_ptr);

    using LoadT = AlignedVector<T, vec_size>;
    LoadT load_vec;
    LoadT res_vec;
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2*)(&res_vec) + i) = make_half2(0, 0);
      }
    } else {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162*)(&res_vec) + i) = make_bfloat162(0, 0);
      }
    }
    float m;
    float d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      m = -3.0e+30f;
    }
#pragma unroll 2
    for (int i = ty; i < num_chunks_this_seq; i += bdy) {
      uint32_t offset;

      offset = ((bid * max_tokens_per_batch + local_seq_id) * num_chunks + i) *
                   num_heads +
               hid;
      float m_prev = m;
      float d_prev = d;
      const float m_now = multi_m[offset];
      const float d_now = multi_d[offset];
      m = max(m_prev, m_now);

      offset = ((bid * max_tokens_per_batch + local_seq_id) * num_chunks *
                    num_heads +
                i * num_heads + hid) *
                   head_dim +
               vid * vec_size;
      Load<T, vec_size>(&multi_out[offset], &load_vec);
      const float scale1 = expf(m_prev - m), scale2 = expf(m_now - m);
      const T scale1_T = static_cast<T>(scale1),
              scale2_T = static_cast<T>(scale2);
      d = d * scale1 + d_now * scale2;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
      }
    }
    // store ty res
    Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
    md_smem[2 * ty] = m;
    md_smem[2 * ty + 1] = d;
    __syncthreads();
    if (ty == 0) {
      // merge bdy
      prefill_softmax_state_t<vec_size, T> st;
      st.init();
#pragma unroll
      for (int i = 0; i < bdy; i++) {
        Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
        const float m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
        st.merge(load_vec, m_tmp, d_tmp);
      }

      if (sinks) {
        float current_sink = static_cast<float>(sinks[hid]);
        st.normalize(current_sink);
      } else {
        st.normalize();
      }

      const uint32_t shift_smooth_offset = hid * head_dim + vid * vec_size;
      AlignedVector<T, vec_size> shift_bias_vec;
      AlignedVector<T, vec_size> smooth_weight_vec;
      AlignedVector<OutT, vec_size> out_vec;
      if (shift_bias) {
        Load<T, vec_size>(shift_bias + shift_smooth_offset, &shift_bias_vec);
        Load<T, vec_size>(smooth_weight + shift_smooth_offset,
                          &smooth_weight_vec);
      }

#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        StoreFunc<T, vec_size, OutT>()(st.o,
                                       shift_bias_vec,
                                       smooth_weight_vec,
                                       out_vec,
                                       quant_max_bound,
                                       quant_min_bound,
                                       in_scale,
                                       i);
      }
      Store<OutT, vec_size>(
          out_vec, &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
    }
    __syncthreads();
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}
