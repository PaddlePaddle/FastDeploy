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

template <typename T>
struct Converter;

template <>
struct Converter<__half> {
  // __half -> float
  __device__ static float to_float(__half val) { return __half2float(val); }
  // float -> __half
  __device__ static __half from_float(float val) {
    return __float2half_rn(val);
  }
  // int -> __half
  __device__ static __half from_int(float val) { return __int2half_rn(val); }
};

template <>
struct Converter<__nv_bfloat16> {
  // __nv_bfloat16 -> float
  __device__ static float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
  }
  // float -> __nv_bfloat16
  __device__ static __nv_bfloat16 from_float(float val) {
    return __float2bfloat16_rn(val);
  }
  // int -> __nv_bfloat16
  __device__ static __nv_bfloat16 from_int(int val) {
    return __int2bfloat16_rn(val);
  }
};

struct CacheKVWithRopeParams {
  int head_dim;
  int block_size;
  int block_num;
  int cache_stride;
  int token_stride;
  int head_stride;
  int q_stride;
  int kv_stride;
  int q_head_offset;
  int k_head_offset;
  int v_head_offset;
  int q_head_num;
  int kv_head_num;
};

template <typename T, int VecSize = 4, bool WriteCache = true>
__device__ __forceinline__ void RotateQKVec(const T* qkv_ptr,
                                            const T* rotary_cos_ptr,
                                            const T* rotary_sin_ptr,
                                            const int load_idx,
                                            const int store_idx,
                                            const int cache_store_idx,
                                            const int rot_base_idx,
                                            T* caches,
                                            T* out) {
  using VecT = AlignedVector<T, VecSize>;

  VecT qk_vec;
  Load(qkv_ptr + load_idx, &qk_vec);
  VecT rot_half_vec;
  int flag;
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    flag = 1 - 2 * (i % 2);
    rot_half_vec[i] = -qk_vec[i + flag] * Converter<T>::from_int(flag);
  }
  VecT cos_vec, sin_vec;
  Load(rotary_cos_ptr + rot_base_idx, &cos_vec);
  Load(rotary_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    T result = qk_vec[i] * cos_vec[i] + rot_half_vec[i] * sin_vec[i];
    *(out + store_idx + i) = result;

    if (WriteCache) {
      *(caches + cache_store_idx + i) = result;
    }
  }
}

template <typename T, int VecSize = 4, bool WriteCache = true>
__device__ __forceinline__ void RotateQKVec(const T* qkv_ptr,
                                            const float* rotary_cos_ptr,
                                            const float* rotary_sin_ptr,
                                            const int load_idx,
                                            const int store_idx,
                                            const int cache_store_idx,
                                            const int rot_base_idx,
                                            T* caches,
                                            T* out) {
  using VecT = AlignedVector<T, VecSize>;
  using VecF = AlignedVector<float, VecSize>;
  auto to_float = [] __device__(T val) -> float {
    return Converter<T>::to_float(val);
  };
  auto from_float = [] __device__(float val) -> T {
    return Converter<T>::from_float(val);
  };

  VecT qk_vec;
  Load(qkv_ptr + load_idx, &qk_vec);
  VecF rot_half_vec;
  int flag;
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    flag = 1 - 2 * (i % 2);
    rot_half_vec[i] = -to_float(qk_vec[i + flag]) * static_cast<float>(flag);
  }
  VecF cos_vec, sin_vec;
  Load(rotary_cos_ptr + rot_base_idx, &cos_vec);
  Load(rotary_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    T result = from_float(to_float(qk_vec[i]) * cos_vec[i] +
                          rot_half_vec[i] * sin_vec[i]);
    *(out + store_idx + i) = result;
    if (WriteCache) {
      *(caches + cache_store_idx + i) = result;
    }
  }
}

template <typename T, int VecSize = 4>
__device__ __forceinline__ void StoreValue(const T* qkv_ptr,
                                           const int load_idx,
                                           const int store_idx,
                                           const int cache_store_idx,
                                           T* caches,
                                           T* out) {
  using VecT = AlignedVector<T, VecSize>;
  VecT v_vec;
  Load(qkv_ptr + load_idx, &v_vec);
  Store(v_vec, out + store_idx);
  Store(v_vec, caches + cache_store_idx);
}

template <typename T, typename WeightType, int VecSize>
__global__ void DispatchCacheKVWithRopeVecKernel(const T* qkv,
                                                 T* caches_k,
                                                 T* caches_v,
                                                 const int* block_tables,
                                                 const WeightType* rotary_cos,
                                                 const WeightType* rotary_sin,
                                                 const int* cu_seqlens_q,
                                                 const int* batch_ids_q,
                                                 CacheKVWithRopeParams param,
                                                 T* q_out,
                                                 T* k_out,
                                                 T* v_out) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int head_dim_idx = (blockIdx.z * blockDim.z + threadIdx.z) * VecSize;

  int load_idx, store_idx, cache_store_idx;
  int rot_idx = token_idx * param.head_dim + head_dim_idx;

  const int batch_idx = *(batch_ids_q + token_idx);
  const int inter_batch_token_offset = token_idx - *(cu_seqlens_q + batch_idx);
  const int inter_batch_block_idx = inter_batch_token_offset / param.block_size;
  const int inter_block_offset = inter_batch_token_offset % param.block_size;
  const int block_idx =
      *(block_tables + batch_idx * param.block_num + inter_batch_block_idx);

  assert(block_idx != -1);

  if (head_dim_idx < param.head_dim) {
    if (head_idx < param.q_head_num) {  // q
      load_idx = token_idx * param.token_stride +
                 (head_idx + param.q_head_offset) * param.head_stride +
                 head_dim_idx;
      store_idx =
          token_idx * param.q_stride + head_idx * param.head_dim + head_dim_idx;
      RotateQKVec<T, VecSize, false>(qkv,
                                     rotary_cos,
                                     rotary_sin,
                                     load_idx,
                                     store_idx,
                                     -1,
                                     rot_idx,
                                     static_cast<T*>(nullptr),
                                     q_out);
    }

    if (head_idx < param.kv_head_num) {  // kv
      load_idx = token_idx * param.token_stride +
                 (head_idx + param.k_head_offset) * param.head_stride +
                 head_dim_idx;
      store_idx = token_idx * param.kv_stride + head_idx * param.head_dim +
                  head_dim_idx;
      cache_store_idx = block_idx * param.cache_stride +
                        inter_block_offset * param.kv_stride +
                        head_idx * param.head_dim + head_dim_idx;
      // printf("block_idx: %d inter_block_offset: %d cache_store_idx: %d
      // param.cache_stride: %d\n", block_idx, inter_block_offset,
      // cache_store_idx, param.cache_stride);
      RotateQKVec<T, VecSize, true>(qkv,
                                    rotary_cos,
                                    rotary_sin,
                                    load_idx,
                                    store_idx,
                                    cache_store_idx,
                                    rot_idx,
                                    caches_k,
                                    k_out);

      load_idx = token_idx * param.token_stride +
                 (head_idx + param.v_head_offset) * param.head_stride +
                 head_dim_idx;
      StoreValue<T, VecSize>(
          qkv, load_idx, store_idx, cache_store_idx, caches_v, v_out);
    }
  }
}

template <paddle::DataType D, int VecSize = 4>
void CacheKVWithRopeKernel(
    const paddle::Tensor& qkv,  // token_num, head_num * head_dim
    paddle::Tensor&
        caches_k,  // max_block_num, block_size, kv_head_num, head_dim
    paddle::Tensor&
        caches_v,  // max_block_num, block_size, kv_head_num, head_dim
    const paddle::Tensor& block_tables,  // bs, block_num
    const paddle::Tensor& rotary_cos,
    const paddle::Tensor& rotary_sin,
    const paddle::Tensor& cu_seqlens_q,  // bs + 1
    const paddle::Tensor& batch_ids_q,   // token_num
    const int q_head_num,
    const int kv_head_num,
    const int head_dim,
    const int block_size,
    paddle::Tensor& q_out,
    paddle::Tensor& k_out,
    paddle::Tensor& v_out) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int all_num_elements = qkv.numel();
  const int all_num_heads = q_head_num + 2 * kv_head_num;
  auto stream = qkv.stream();

  dim3 block_dims(1, 4, (head_dim + VecSize - 1) / VecSize);
  dim3 grid_dims(all_num_elements / (all_num_heads * head_dim),  // token
                 (std::max(q_head_num, kv_head_num) + block_dims.y - 1) /
                     block_dims.y,  // head
                 (head_dim + (block_dims.z * VecSize) - 1) /
                     (block_dims.z * VecSize)  // dim: load Vec at a time
  );

  // printf("grid: (%d, %d, %d)\n", grid_dims.x, grid_dims.y, grid_dims.z);
  // printf("block: (%d, %d, %d)\n", block_dims.x, block_dims.y, block_dims.z);

  CacheKVWithRopeParams param;
  param.head_dim = head_dim;
  param.block_size = block_size;
  param.block_num = static_cast<int>(block_tables.shape().back());
  param.cache_stride = block_size * kv_head_num * head_dim;
  param.token_stride = all_num_heads * head_dim;
  param.head_stride = head_dim;
  param.q_stride = q_head_num * head_dim;
  param.kv_stride = kv_head_num * head_dim;
  param.q_head_offset = 0;
  param.k_head_offset = q_head_num;
  param.v_head_offset = q_head_num + kv_head_num;
  param.q_head_num = q_head_num;
  param.kv_head_num = kv_head_num;

  if (qkv.dtype() == rotary_cos.dtype()) {
    DispatchCacheKVWithRopeVecKernel<DataType_, DataType_, VecSize>
        <<<grid_dims, block_dims, 0, stream>>>(
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<DataType_*>(caches_k.data<data_t>()),
            reinterpret_cast<DataType_*>(caches_v.data<data_t>()),
            reinterpret_cast<const int*>(block_tables.data<int>()),
            reinterpret_cast<const DataType_*>(rotary_cos.data<data_t>()),
            reinterpret_cast<const DataType_*>(rotary_sin.data<data_t>()),
            reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),
            reinterpret_cast<const int*>(batch_ids_q.data<int>()),
            param,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(v_out.data<data_t>()));
  } else if (rotary_cos.dtype() == paddle::DataType::FLOAT32) {
    DispatchCacheKVWithRopeVecKernel<DataType_, float, VecSize>
        <<<grid_dims, block_dims, 0, stream>>>(
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<DataType_*>(caches_k.data<data_t>()),
            reinterpret_cast<DataType_*>(caches_v.data<data_t>()),
            reinterpret_cast<const int*>(block_tables.data<int>()),
            reinterpret_cast<const float*>(rotary_cos.data<float>()),
            reinterpret_cast<const float*>(rotary_sin.data<float>()),
            reinterpret_cast<const int*>(cu_seqlens_q.data<int>()),
            reinterpret_cast<const int*>(batch_ids_q.data<int>()),
            param,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(v_out.data<data_t>()));
  } else {
    PD_THROW("Unsupported qk dtype and rope dtype.");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}

std::vector<paddle::Tensor> CacheKVWithRope(
    const paddle::Tensor& qkv,  // token_num, head_num * head_dim
    paddle::Tensor&
        caches_k,  // max_block_num, block_size, kv_head_num, head_dim
    paddle::Tensor&
        caches_v,  // max_block_num, block_size, kv_head_num, head_dim
    const paddle::Tensor& block_tables,  // bs, block_num
    const paddle::Tensor& rotary_cos,
    const paddle::Tensor& rotary_sin,
    const paddle::Tensor& cu_seqlens_q,  // bs + 1
    const paddle::Tensor& batch_ids_q,   // token_num
    const int q_head_num,
    const int kv_head_num,
    const int head_dim,
    const int block_size) {
  auto qkv_shape = qkv.shape();
  auto token_num = qkv_shape[0];
  auto place = qkv.place();
  auto dtype = qkv.dtype();
  common::DDim q_out_shape, kv_out_shape;
  if (rotary_cos.shape().size() == 3) {
    q_out_shape = {token_num, q_head_num, head_dim};
    kv_out_shape = {token_num, kv_head_num, head_dim};
  } else {
    q_out_shape = {token_num, 1, q_head_num, head_dim};
    kv_out_shape = {token_num, 1, kv_head_num, head_dim};
  }
  auto q_out = GetEmptyTensor(q_out_shape, dtype, place);
  auto k_out = GetEmptyTensor(kv_out_shape, dtype, place);
  auto v_out = GetEmptyTensor(kv_out_shape, dtype, place);

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
  PADDLE_ENFORCE_EQ(q_out.shape().back(),
                    rotary_cos.shape().back(),
                    "The last dimension of cos mismatches that of q, "
                    "expect %d but got %d",
                    q_out.shape().back(),
                    rotary_cos.shape().back());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      CacheKVWithRopeKernel<paddle::DataType::BFLOAT16>(qkv,
                                                        caches_k,
                                                        caches_v,
                                                        block_tables,
                                                        rotary_cos,
                                                        rotary_sin,
                                                        cu_seqlens_q,
                                                        batch_ids_q,
                                                        q_head_num,
                                                        kv_head_num,
                                                        head_dim,
                                                        block_size,
                                                        q_out,
                                                        k_out,
                                                        v_out);
      break;
    case paddle::DataType::FLOAT16:
      CacheKVWithRopeKernel<paddle::DataType::FLOAT16>(qkv,
                                                       caches_k,
                                                       caches_v,
                                                       block_tables,
                                                       rotary_cos,
                                                       rotary_sin,
                                                       cu_seqlens_q,
                                                       batch_ids_q,
                                                       q_head_num,
                                                       kv_head_num,
                                                       head_dim,
                                                       block_size,
                                                       q_out,
                                                       k_out,
                                                       v_out);
      break;
    default:
      PD_THROW("Only support qk dtype of BF16 and F16");
  }

  return {q_out, k_out, v_out};
}

std::vector<std::vector<int64_t>> CacheKVWithRopeInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& caches_k_shape,
    const std::vector<int64_t>& caches_v_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& cos_shape,
    const std::vector<int64_t>& sin_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& batch_ids_q_shape) {
  return {qkv_shape,
          caches_k_shape,
          caches_v_shape,
          block_tables_shape,
          cos_shape,
          sin_shape,
          cu_seqlens_q_shape,
          batch_ids_q_shape};
}

std::vector<paddle::DataType> CacheKVWithRopeInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& caches_k_dtype,
    const paddle::DataType& caches_v_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& cos_dtype,
    const paddle::DataType& sin_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& batch_ids_q_dtype) {
  return {qkv_dtype,
          caches_k_dtype,
          caches_v_dtype,
          block_tables_dtype,
          cos_dtype,
          sin_dtype,
          cu_seqlens_q_dtype,
          batch_ids_q_dtype};
}

PD_BUILD_OP(cache_kv_with_rope)
    .Inputs({"qkv",
             "caches_k",
             "caches_v",
             "block_tables",
             "rotary_cos",
             "rotary_sin",
             "cu_seqlen_q",
             "batch_ids_q"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs(
        {"q_head_num:int", "kv_head_num:int", "head_dim:int", "block_size:int"})
    .SetKernelFn(PD_KERNEL(CacheKVWithRope))
    .SetInferShapeFn(PD_INFER_SHAPE(CacheKVWithRopeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CacheKVWithRopeInferDtype));
