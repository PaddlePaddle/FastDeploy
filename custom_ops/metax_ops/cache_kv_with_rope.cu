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

#include <cuda_runtime.h>
#include <paddle/extension.h>
#include <algorithm>
#include "helper.h"

struct CacheKVWithRopeParams {
  int64_t linear_elem_num;
  int linear_stride;
  int head_dim;
  int block_size;
  int block_num;
  int cache_stride;
  int token_stride;
  int q_stride;
  int kv_stride;
  int k_head_offset;
  int v_head_offset;
  int q_head_num;
  int kv_head_num;
  int rotary_stride;        // 1 * S * 1 * D / 2 or 1 * S * 1 * D(if neox)
  int batch_rotary_stride;  // 2 * rotary_stride
};

template <typename T, int VecSize, bool WriteCache>
__device__ __forceinline__ void RotateQKVec(
    const T* __restrict__ qkv_ptr,
    const float* __restrict__ rotary_embs_ptr,
    const int load_idx,
    const int store_idx,
    const int cache_store_idx,
    T* __restrict__ caches,
    T* __restrict__ out) {
  using VecQKV = AlignedVector<T, VecSize>;
  const int SIN_OFFSET = VecSize / 2;

  VecQKV qk_vec, qk_out_vec;
  Load(qkv_ptr + load_idx, &qk_vec);

#pragma unroll
  for (int i = 0; i < VecSize; i += 2) {
    float q0 = static_cast<float>(qk_vec[i]);
    float q1 = static_cast<float>(qk_vec[i + 1]);

    float cos_val = rotary_embs_ptr[i >> 1];
    float sin_val = rotary_embs_ptr[SIN_OFFSET + (i >> 1)];

    qk_out_vec[i] = static_cast<T>(q0 * cos_val - q1 * sin_val);
    qk_out_vec[i + 1] = static_cast<T>(q1 * cos_val + q0 * sin_val);
  }

  Store(qk_out_vec, out + store_idx);
  if constexpr (WriteCache) {
    Store(qk_out_vec, caches + cache_store_idx);
  }
}

template <typename T, int VecSize, bool WriteCache>
__device__ __forceinline__ void StoreValue(const T* __restrict__ qkv_ptr,
                                           const int load_idx,
                                           const int store_idx,
                                           const int cache_store_idx,
                                           T* __restrict__ caches,
                                           T* __restrict__ out) {
  using VecT = AlignedVector<T, VecSize>;
  VecT v_vec;
  Load(qkv_ptr + load_idx, &v_vec);
  Store(v_vec, out + store_idx);
  if constexpr (WriteCache) {
    Store(v_vec, caches + cache_store_idx);
  }
}

template <typename T, int VecSize, bool WriteCache>
__device__ __forceinline__ void RotateQKVecNeox(
    const T* __restrict__ qkv_ptr,
    const float* __restrict__ rotary_embs_ptr,
    const int left_load_idx,
    const int right_load_idx,
    const int left_store_idx,
    const int right_store_idx,
    const int left_cache_store_idx,
    const int right_cache_store_idx,
    T* __restrict__ caches,
    T* __restrict__ out) {
  using VecQKV = AlignedVector<T, VecSize>;
  constexpr int SIN_OFFSET = VecSize;

  VecQKV left_vec, right_vec, left_out_vec, right_out_vec;

  Load(qkv_ptr + left_load_idx, &left_vec);
  Load(qkv_ptr + right_load_idx, &right_vec);

#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    float l_val = static_cast<float>(left_vec[i]);
    float r_val = static_cast<float>(right_vec[i]);

    float cos_val = rotary_embs_ptr[i];
    float sin_val = rotary_embs_ptr[SIN_OFFSET + i];

    left_out_vec[i] = static_cast<T>(l_val * cos_val - r_val * sin_val);
    right_out_vec[i] = static_cast<T>(r_val * cos_val + l_val * sin_val);
  }

  Store(left_out_vec, out + left_store_idx);
  Store(right_out_vec, out + right_store_idx);

  if constexpr (WriteCache) {
    Store(left_out_vec, caches + left_cache_store_idx);
    Store(right_out_vec, caches + right_cache_store_idx);
  }
}

template <typename T, int VecSize, bool WriteCache>
__device__ __forceinline__ void StoreValueNeox(const T* __restrict__ qkv_ptr,
                                               const int left_load_idx,
                                               const int right_load_idx,
                                               const int left_store_idx,
                                               const int right_store_idx,
                                               const int left_cache_store_idx,
                                               const int right_cache_store_idx,
                                               T* __restrict__ caches,
                                               T* __restrict__ out) {
  using VecT = AlignedVector<T, VecSize>;
  VecT left_v_vec, right_v_vec;
  Load(qkv_ptr + left_load_idx, &left_v_vec);
  Load(qkv_ptr + right_load_idx, &right_v_vec);
  Store(left_v_vec, out + left_store_idx);
  Store(right_v_vec, out + right_store_idx);
  if constexpr (WriteCache) {
    Store(left_v_vec, caches + left_cache_store_idx);
    Store(right_v_vec, caches + right_cache_store_idx);
  }
}

struct CacheKVIndices {
  // 线程块索引
  int token_idx;
  int head_idx;
  int head_dim_idx;

  // RoPE 旋转索引
  int rotary_cos_idx;
  int rotary_sin_idx;

  // 全局内存 Load/Store 索引
  int load_idx[3];   // q, k, v
  int store_idx[3];  // q, kv

  // KV Cache 存储索引 (根据模板参数计算，但自身仍是 int 类型)
  int cache_store_idx;
  int right_cache_store_idx;
};

// 辅助函数：计算所有索引
template <bool WriteCache, bool NeoxStyle>
__device__ void GetIndices(int64_t linear_index,
                           const int half_head_dim,
                           const int* __restrict__ batch_ids_per_token,
                           const int* __restrict__ global_batch_ids,
                           const int* __restrict__ cu_seqlens_q,
                           const int* __restrict__ seqlens_q,
                           const int* __restrict__ block_tables,
                           const CacheKVWithRopeParams& param,
                           CacheKVIndices& indices) {
  // ********** 1. linear index -> 3D index **********
  if constexpr (NeoxStyle) {
    int linear_stride_half = (param.linear_stride >> 1);
    int head_dim_half = (param.head_dim >> 1);
    indices.token_idx = linear_index / linear_stride_half;
    indices.head_idx = (linear_index % linear_stride_half) / head_dim_half;
    indices.head_dim_idx = linear_index % head_dim_half;
  } else {
    indices.token_idx = linear_index / param.linear_stride;
    indices.head_idx = (linear_index % param.linear_stride) / param.head_dim;
    indices.head_dim_idx = linear_index % param.head_dim;
  }

  // ********** 2. QKV Load Index **********
  indices.load_idx[0] = indices.token_idx * param.token_stride +
                        indices.head_idx * param.head_dim +
                        indices.head_dim_idx;
  indices.load_idx[1] =
      indices.load_idx[0] + param.k_head_offset * param.head_dim;
  indices.load_idx[2] =
      indices.load_idx[0] + param.v_head_offset * param.head_dim;

  // ********** 3. Batch and Seq Index **********
  const int local_batch_idx = *(batch_ids_per_token + indices.token_idx);
  const int global_batch_idx = *(global_batch_ids + local_batch_idx);
  const int inter_batch_token_offset = indices.token_idx +
                                       *(seqlens_q + local_batch_idx) -
                                       *(cu_seqlens_q + local_batch_idx);

  // ********** 4. RoPE Index **********
  if constexpr (!NeoxStyle) {
    indices.rotary_cos_idx = global_batch_idx * param.batch_rotary_stride +
                             inter_batch_token_offset * half_head_dim;
  } else {
    indices.rotary_cos_idx = global_batch_idx * param.batch_rotary_stride +
                             inter_batch_token_offset * param.head_dim;
  }

  if constexpr (!NeoxStyle) {
    indices.rotary_cos_idx += (indices.head_dim_idx >> 1);
  } else {
    indices.rotary_cos_idx += indices.head_dim_idx % half_head_dim;
  }
  indices.rotary_sin_idx = indices.rotary_cos_idx + param.rotary_stride;

  // ********** 5. QKV Store Index **********
  indices.store_idx[0] = indices.token_idx * param.q_stride +
                         indices.head_idx * param.head_dim +
                         indices.head_dim_idx;
  indices.store_idx[1] = indices.token_idx * param.kv_stride +
                         indices.head_idx * param.head_dim +
                         indices.head_dim_idx;
  indices.store_idx[2] = indices.store_idx[1];

  // ********** 6. KV Cache Store Index (仅 WriteCache) **********
  indices.cache_store_idx = -1;
  indices.right_cache_store_idx = -1;

  if constexpr (WriteCache) {
    const int inter_batch_block_idx =
        inter_batch_token_offset / param.block_size;
    const int inter_block_offset = inter_batch_token_offset % param.block_size;
    const int block_idx = *(block_tables + global_batch_idx * param.block_num +
                            inter_batch_block_idx);

    assert(block_idx != -1);

    indices.cache_store_idx =
        block_idx * param.cache_stride + inter_block_offset * param.kv_stride +
        indices.head_idx * param.head_dim + indices.head_dim_idx;

    if constexpr (NeoxStyle) {
      indices.right_cache_store_idx = indices.cache_store_idx + half_head_dim;
    }
  }
}

template <typename WeightType, int RotaryVecSize>
__device__ inline void preload_rotary(
    const WeightType* __restrict__ rotary_embs,
    const int rotary_cos_idx,
    const int rotary_sin_idx,
    float* __restrict__ rotary_embs_vec) {
  using VecRotary = AlignedVector<float, RotaryVecSize>;

  VecRotary* rotary_cos_vec = reinterpret_cast<VecRotary*>(rotary_embs_vec);
  VecRotary* rotary_sin_vec =
      reinterpret_cast<VecRotary*>(rotary_embs_vec + RotaryVecSize);

  if constexpr (std::is_same_v<WeightType, float>) {
    Load(rotary_embs + rotary_cos_idx, rotary_cos_vec);
    Load(rotary_embs + rotary_sin_idx, rotary_sin_vec);
  } else {
#pragma unroll
    for (int i = 0; i < RotaryVecSize; ++i) {
      (*rotary_cos_vec)[i] =
          static_cast<float>(__ldg(rotary_embs + rotary_cos_idx + i));
      (*rotary_sin_vec)[i] =
          static_cast<float>(__ldg(rotary_embs + rotary_sin_idx + i));
    }
  }
}

template <typename T,
          typename WeightType,
          int VecSize,
          bool WriteCache,
          bool NeoxStyle>
__global__ void DispatchCacheKVWithRopeVecKernel(
    const T* __restrict__ qkv,
    const WeightType* __restrict__ rotary_embs,
    const int* __restrict__ batch_ids_per_token,
    const int* __restrict__ global_batch_ids,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seqlens_q,
    T* __restrict__ caches_k,
    T* caches_v,
    const int* __restrict__ block_tables,
    CacheKVWithRopeParams param,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out) {
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_head_dim = (param.head_dim >> 1);
  int64_t max_elements =
      NeoxStyle ? (param.linear_elem_num >> 1) : param.linear_elem_num;

  constexpr int VecRotarySize2 = NeoxStyle ? VecSize * 2 : VecSize;
  using VecRotary2 = AlignedVector<WeightType, VecRotarySize2>;
  VecRotary2 rotary_embs_vec;
  float* rotary_embs_vec_ptr = reinterpret_cast<float*>(&rotary_embs_vec);

  // Grid Stride Loop
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = (int64_t)gridDim.x * blockDim.x * VecSize;
       linear_index < max_elements;
       linear_index += step) {
    // ********** 索引计算 **********
    CacheKVIndices indices;
    GetIndices<WriteCache, NeoxStyle>(linear_index,
                                      half_head_dim,
                                      batch_ids_per_token,
                                      global_batch_ids,
                                      cu_seqlens_q,
                                      seqlens_q,
                                      block_tables,
                                      param,
                                      indices);

    preload_rotary<WeightType, VecRotarySize2 / 2>(rotary_embs,
                                                   indices.rotary_cos_idx,
                                                   indices.rotary_sin_idx,
                                                   rotary_embs_vec_ptr);

    if (indices.head_idx < param.q_head_num) {
      // ********** 1. Q 向量旋转与存储 **********
      if constexpr (!NeoxStyle) {
        RotateQKVec<T, VecSize, false>(qkv,
                                       rotary_embs_vec_ptr,
                                       indices.load_idx[0],
                                       indices.store_idx[0],
                                       -1,
                                       static_cast<T*>(nullptr),
                                       q_out);
      } else {
        int right_load_idx = indices.load_idx[0] + half_head_dim;
        int right_store_idx = indices.store_idx[0] + half_head_dim;
        RotateQKVecNeox<T, VecSize, false>(qkv,
                                           rotary_embs_vec_ptr,
                                           indices.load_idx[0],
                                           right_load_idx,
                                           indices.store_idx[0],
                                           right_store_idx,
                                           -1,
                                           -1,
                                           static_cast<T*>(nullptr),
                                           q_out);
      }
    }

    if (indices.head_idx < param.kv_head_num) {
      // ********** 2. K 向量旋转与存储/缓存 **********
      if constexpr (!NeoxStyle) {
        RotateQKVec<T, VecSize, WriteCache>(qkv,
                                            rotary_embs_vec_ptr,
                                            indices.load_idx[1],
                                            indices.store_idx[1],
                                            indices.cache_store_idx,
                                            caches_k,
                                            k_out);
      } else {
        int right_load_idx = indices.load_idx[1] + half_head_dim;
        int right_store_idx = indices.store_idx[1] + half_head_dim;
        RotateQKVecNeox<T, VecSize, WriteCache>(qkv,
                                                rotary_embs_vec_ptr,
                                                indices.load_idx[1],
                                                right_load_idx,
                                                indices.store_idx[1],
                                                right_store_idx,
                                                indices.cache_store_idx,
                                                indices.right_cache_store_idx,
                                                caches_k,
                                                k_out);
      }

      // ********** 3. V 向量直通与存储/缓存 **********
      if constexpr (!NeoxStyle) {
        StoreValue<T, VecSize, WriteCache>(qkv,
                                           indices.load_idx[2],
                                           indices.store_idx[2],
                                           indices.cache_store_idx,
                                           caches_v,
                                           v_out);
      } else {
        int right_load_idx = indices.load_idx[2] + half_head_dim;
        int right_store_idx = indices.store_idx[2] + half_head_dim;
        StoreValueNeox<T, VecSize, WriteCache>(qkv,
                                               indices.load_idx[2],
                                               right_load_idx,
                                               indices.store_idx[2],
                                               right_store_idx,
                                               indices.cache_store_idx,
                                               indices.right_cache_store_idx,
                                               caches_v,
                                               v_out);
      }
    }
  }
}

template <paddle::DataType D, int VecSize>
void CacheKVWithRopeKernel(
    const paddle::Tensor& qkv,  // token_num, head_num * head_dim
    const paddle::Tensor&
        rotary_embs,  // [2, 1, max_seqlens, 1, half_head_dim(head_dim if neox)]
                      // or [bs, 2, 1, max_seqlens, 1, half_head_dim(head_dim if
                      // neox)]
    const paddle::Tensor& batch_ids_per_token,  // token_num
    const paddle::Tensor& global_batch_ids,
    const paddle::Tensor& cu_seqlens_q,  // bs + 1
    const paddle::Tensor& seqlens_q,     // bs
    paddle::optional<paddle::Tensor>&
        caches_k,  // max_block_num, block_size, kv_head_num, head_dim
    paddle::optional<paddle::Tensor>&
        caches_v,  // max_block_num, block_size, kv_head_num, head_dim
    const paddle::optional<paddle::Tensor>& block_tables,  // bs, block_num
    const int q_head_num,
    const int kv_head_num,
    const int head_dim,
    const int block_size,
    const bool neox_style,
    paddle::Tensor& q_out,
    paddle::Tensor& k_out,
    paddle::Tensor& v_out) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int64_t linear_elem_num =
      qkv.shape()[0] * std::max(q_head_num, kv_head_num) * head_dim;
  const int all_num_heads = q_head_num + 2 * kv_head_num;
  auto stream = qkv.stream();

  const int pack_size = neox_style ? (VecSize * 2) : VecSize;
  const int pack_num = linear_elem_num / pack_size;
  const int block_dims = 128;
  int grid_dims = 1;
  GetNumBlocks<block_dims>(pack_num, &grid_dims);

  // printf("grid: (%d, %d, %d)\n", grid_dims.x, grid_dims.y, grid_dims.z);
  // printf("block: (%d, %d, %d)\n", block_dims.x, block_dims.y, block_dims.z);

  CacheKVWithRopeParams param;
  param.linear_elem_num = linear_elem_num;
  param.linear_stride = std::max(q_head_num, kv_head_num) * head_dim;
  param.head_dim = head_dim;
  param.block_size = block_size;
  if (block_tables) {
    param.block_num = static_cast<int>(block_tables.get_ptr()->shape().back());
  } else {
    param.block_num = -1;
  }
  param.cache_stride = block_size * kv_head_num * head_dim;
  param.token_stride = all_num_heads * head_dim;
  param.q_stride = q_head_num * head_dim;
  param.kv_stride = kv_head_num * head_dim;
  param.k_head_offset = q_head_num;
  param.v_head_offset = q_head_num + kv_head_num;
  param.q_head_num = q_head_num;
  param.kv_head_num = kv_head_num;
  const auto rotary_embs_shape = rotary_embs.shape();
  if (!neox_style) {
    if (rotary_embs_shape.size() == 5) {
      param.rotary_stride = rotary_embs_shape[2] * head_dim / 2;
      param.batch_rotary_stride = 0;
    } else {
      param.rotary_stride = rotary_embs_shape[3] * head_dim / 2;
      param.batch_rotary_stride = 2 * param.rotary_stride;
    }
  } else {
    if (rotary_embs_shape.size() == 5) {
      param.rotary_stride = rotary_embs_shape[2] * head_dim;
      param.batch_rotary_stride = 0;
    } else {
      param.rotary_stride = rotary_embs_shape[3] * head_dim;
      param.batch_rotary_stride = 2 * param.rotary_stride;
    }
  }

#define APPLY_ROPE_AND_WRITE_CACHE(DATATYPE, DATA_T)                          \
  DispatchCacheKVWithRopeVecKernel<DataType_, DATATYPE, VecSize, true, false> \
      <<<grid_dims, block_dims, 0, stream>>>(                                 \
          reinterpret_cast<const DataType_*>(qkv.data<data_t>()),             \
          reinterpret_cast<const DATATYPE*>(rotary_embs.data<DATA_T>()),      \
          reinterpret_cast<const int*>(batch_ids_per_token.data<int>()),      \
          reinterpret_cast<const int*>(global_batch_ids.data<int>()),         \
          reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),             \
          reinterpret_cast<const int*>(seqlens_q.data<int>()),                \
          reinterpret_cast<DataType_*>(caches_k.get_ptr()->data<data_t>()),   \
          reinterpret_cast<DataType_*>(caches_v.get_ptr()->data<data_t>()),   \
          reinterpret_cast<const int*>(block_tables.get_ptr()->data<int>()),  \
          param,                                                              \
          reinterpret_cast<DataType_*>(q_out.data<data_t>()),                 \
          reinterpret_cast<DataType_*>(k_out.data<data_t>()),                 \
          reinterpret_cast<DataType_*>(v_out.data<data_t>()));

#define APPLY_ROPE(DATATYPE, DATA_T)                                           \
  DispatchCacheKVWithRopeVecKernel<DataType_, DATATYPE, VecSize, false, false> \
      <<<grid_dims, block_dims, 0, stream>>>(                                  \
          reinterpret_cast<const DataType_*>(qkv.data<data_t>()),              \
          reinterpret_cast<const DATATYPE*>(rotary_embs.data<DATA_T>()),       \
          reinterpret_cast<const int*>(batch_ids_per_token.data<int>()),       \
          reinterpret_cast<const int*>(global_batch_ids.data<int>()),          \
          reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),              \
          reinterpret_cast<const int*>(seqlens_q.data<int>()),                 \
          static_cast<DataType_*>(nullptr),                                    \
          static_cast<DataType_*>(nullptr),                                    \
          static_cast<const int*>(nullptr),                                    \
          param,                                                               \
          reinterpret_cast<DataType_*>(q_out.data<data_t>()),                  \
          reinterpret_cast<DataType_*>(k_out.data<data_t>()),                  \
          reinterpret_cast<DataType_*>(v_out.data<data_t>()));

#define APPLY_ROPE_AND_WRITE_CACHE_NEOX(DATATYPE, DATA_T)                    \
  DispatchCacheKVWithRopeVecKernel<DataType_, DATATYPE, VecSize, true, true> \
      <<<grid_dims, block_dims, 0, stream>>>(                                \
          reinterpret_cast<const DataType_*>(qkv.data<data_t>()),            \
          reinterpret_cast<const DATATYPE*>(rotary_embs.data<DATA_T>()),     \
          reinterpret_cast<const int*>(batch_ids_per_token.data<int>()),     \
          reinterpret_cast<const int*>(global_batch_ids.data<int>()),        \
          reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),            \
          reinterpret_cast<const int*>(seqlens_q.data<int>()),               \
          reinterpret_cast<DataType_*>(caches_k.get_ptr()->data<data_t>()),  \
          reinterpret_cast<DataType_*>(caches_v.get_ptr()->data<data_t>()),  \
          reinterpret_cast<const int*>(block_tables.get_ptr()->data<int>()), \
          param,                                                             \
          reinterpret_cast<DataType_*>(q_out.data<data_t>()),                \
          reinterpret_cast<DataType_*>(k_out.data<data_t>()),                \
          reinterpret_cast<DataType_*>(v_out.data<data_t>()));

#define APPLY_ROPE_NEOX(DATATYPE, DATA_T)                                     \
  DispatchCacheKVWithRopeVecKernel<DataType_, DATATYPE, VecSize, false, true> \
      <<<grid_dims, block_dims, 0, stream>>>(                                 \
          reinterpret_cast<const DataType_*>(qkv.data<data_t>()),             \
          reinterpret_cast<const DATATYPE*>(rotary_embs.data<DATA_T>()),      \
          reinterpret_cast<const int*>(batch_ids_per_token.data<int>()),      \
          reinterpret_cast<const int*>(global_batch_ids.data<int>()),         \
          reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),             \
          reinterpret_cast<const int*>(seqlens_q.data<int>()),                \
          static_cast<DataType_*>(nullptr),                                   \
          static_cast<DataType_*>(nullptr),                                   \
          static_cast<const int*>(nullptr),                                   \
          param,                                                              \
          reinterpret_cast<DataType_*>(q_out.data<data_t>()),                 \
          reinterpret_cast<DataType_*>(k_out.data<data_t>()),                 \
          reinterpret_cast<DataType_*>(v_out.data<data_t>()));

#define DISPATCH_CASE(WRITE_CACHE_FUNC, APPLY_ROPE_FUNC)                     \
  if (caches_k && caches_v) {                                                \
    if (qkv.dtype() == rotary_embs.dtype()) {                                \
      WRITE_CACHE_FUNC(DataType_, data_t)                                    \
    } else if (rotary_embs.dtype() == paddle::DataType::FLOAT32) {           \
      WRITE_CACHE_FUNC(float, float)                                         \
    } else {                                                                 \
      PD_THROW(                                                              \
          "qk dtype and rope dtype should be equal or rope dtype is float"); \
    }                                                                        \
  } else {                                                                   \
    if (qkv.dtype() == rotary_embs.dtype()) {                                \
      APPLY_ROPE_FUNC(DataType_, data_t)                                     \
    } else if (rotary_embs.dtype() == paddle::DataType::FLOAT32) {           \
      APPLY_ROPE_FUNC(float, float)                                          \
    } else {                                                                 \
      PD_THROW(                                                              \
          "qk dtype and rope dtype should be equal or rope dtype is float"); \
    }                                                                        \
  }

  if (neox_style) {
    DISPATCH_CASE(APPLY_ROPE_AND_WRITE_CACHE_NEOX, APPLY_ROPE_NEOX)
  } else {
    DISPATCH_CASE(APPLY_ROPE_AND_WRITE_CACHE, APPLY_ROPE)
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}

std::vector<paddle::Tensor> CacheKVWithRope(
    const paddle::Tensor& qkv,  // token_num, head_num * head_dim
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& batch_ids_per_token,
    const paddle::Tensor& global_batch_ids,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& seqlens_q,
    paddle::optional<paddle::Tensor>&
        caches_k,  // max_block_num, block_size, kv_head_num, head_dim
    paddle::optional<paddle::Tensor>&
        caches_v,  // max_block_num, block_size, kv_head_num, head_dim
    const paddle::optional<paddle::Tensor>& block_tables,  // bs, block_num
    const int q_head_num,
    const int kv_head_num,
    const int head_dim,
    const int block_size,
    const int out_dims,
    const bool neox_style) {
  auto qkv_shape = qkv.shape();
  auto token_num = qkv_shape[0];
  auto place = qkv.place();
  auto dtype = qkv.dtype();

  paddle::Tensor q_out, k_out, v_out;
  PD_CHECK(out_dims == 3 || out_dims == 4);
  if (out_dims == 3) {
    q_out = GetEmptyTensor({token_num, q_head_num, head_dim}, dtype, place);
    k_out = GetEmptyTensor({token_num, kv_head_num, head_dim}, dtype, place);
    v_out = GetEmptyTensor({token_num, kv_head_num, head_dim}, dtype, place);
  } else {
    q_out = GetEmptyTensor({token_num, 1, q_head_num, head_dim}, dtype, place);
    k_out = GetEmptyTensor({token_num, 1, kv_head_num, head_dim}, dtype, place);
    v_out = GetEmptyTensor({token_num, 1, kv_head_num, head_dim}, dtype, place);
  }

  if (token_num == 0) {
    return {q_out, k_out, v_out};
  }

  PADDLE_ENFORCE_EQ(qkv_shape.back(),
                    ((q_head_num + 2 * kv_head_num) * head_dim),
                    "The last dimension of qkv [%d] must equal to {(q_head_num "
                    "+ 2 * kv_head_num) * head_dim [%d].",
                    qkv_shape.back(),
                    ((q_head_num + 2 * kv_head_num) * head_dim));
  PADDLE_ENFORCE_EQ(
      head_dim % 2,
      0,
      "The last dimension (head_dim) of qkv must be an even number "
      "for RoPE, but got %d",
      head_dim);
  if (!neox_style) {
    PADDLE_ENFORCE_EQ((q_out.shape().back() / 2),
                      rotary_embs.shape().back(),
                      "The last dimension of cos mismatches that half of q, "
                      "expect %d but got %d",
                      (q_out.shape().back() / 2),
                      rotary_embs.shape().back());
  } else {
    PADDLE_ENFORCE_EQ((q_out.shape().back()),
                      rotary_embs.shape().back(),
                      "The last dimension of cos mismatches that head_dim, "
                      "expect %d but got %d",
                      (q_out.shape().back()),
                      rotary_embs.shape().back());
  }

  if (caches_k && caches_v) {
    if (!block_tables) {
      PD_THROW("block_tables should have value if writing into cache.");
    }
  }

#define KERNEL_CASE(DTYPE)                               \
  case DTYPE:                                            \
    CacheKVWithRopeKernel<DTYPE, 4>(qkv,                 \
                                    rotary_embs,         \
                                    batch_ids_per_token, \
                                    global_batch_ids,    \
                                    cu_seqlens_q,        \
                                    seqlens_q,           \
                                    caches_k,            \
                                    caches_v,            \
                                    block_tables,        \
                                    q_head_num,          \
                                    kv_head_num,         \
                                    head_dim,            \
                                    block_size,          \
                                    neox_style,          \
                                    q_out,               \
                                    k_out,               \
                                    v_out);              \
    break;

  switch (dtype) {
    KERNEL_CASE(paddle::DataType::BFLOAT16)
    KERNEL_CASE(paddle::DataType::FLOAT16)
    default:
      PD_THROW("Only support qk dtype of BF16 and F16");
  }

  return {q_out, k_out, v_out};
}

std::vector<std::vector<int64_t>> CacheKVWithRopeInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& rotary_embs_shape,
    const std::vector<int64_t>& batch_ids_per_token_shape,
    const std::vector<int64_t>& global_batch_ids_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& seqlens_q_shape,
    const paddle::optional<std::vector<int64_t>>& caches_k_shape,
    const paddle::optional<std::vector<int64_t>>& caches_v_shape,
    const paddle::optional<std::vector<int64_t>>& block_tables_shape) {
  return {qkv_shape,
          rotary_embs_shape,
          batch_ids_per_token_shape,
          global_batch_ids_shape,
          cu_seqlens_q_shape,
          seqlens_q_shape};
}

std::vector<paddle::DataType> CacheKVWithRopeInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& rotary_embs_dtype,
    const paddle::DataType& batch_ids_per_token_dtype,
    const paddle::DataType& global_batch_ids_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& seqlens_q_dtype,
    const paddle::optional<paddle::DataType>& caches_k_dtype,
    const paddle::optional<paddle::DataType>& caches_v_dtype,
    const paddle::optional<paddle::DataType>& block_tables_dtype) {
  return {qkv_dtype,
          rotary_embs_dtype,
          batch_ids_per_token_dtype,
          global_batch_ids_dtype,
          cu_seqlens_q_dtype,
          seqlens_q_dtype};
}

PD_BUILD_OP(cache_kv_with_rope)
    .Inputs({
        "qkv",
        "rotary_embs",
        "batch_ids_per_token",
        "global_batch_ids",
        "cu_seqlens_q",
        "seqlens_q",
        paddle::Optional("caches_k"),
        paddle::Optional("caches_v"),
        paddle::Optional("block_tables"),
    })
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"q_head_num:int",
            "kv_head_num:int",
            "head_dim:int",
            "block_size:int",
            "out_dims:int",
            "neox_style:bool"})
    .SetKernelFn(PD_KERNEL(CacheKVWithRope))
    .SetInferShapeFn(PD_INFER_SHAPE(CacheKVWithRopeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CacheKVWithRopeInferDtype));
