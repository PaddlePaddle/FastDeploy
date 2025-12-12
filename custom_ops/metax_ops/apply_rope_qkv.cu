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

struct ApplyRopeQKVParams {
  int head_dim;
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

template <typename T>
__device__ __forceinline__ void RotateQKVec4(const T* qkv_ptr,
                                             const T* rot_cos_ptr,
                                             const T* rot_sin_ptr,
                                             const int load_idx,
                                             const int store_idx,
                                             const int rot_base_idx,
                                             T* out) {
  using VecT = AlignedVector<T, 4>;

  VecT qk_vec;
  Load(qkv_ptr + load_idx, &qk_vec);
  VecT rot_half_vec = {-qk_vec[1], qk_vec[0], -qk_vec[3], qk_vec[2]};
  VecT cos_vec, sin_vec;
  Load(rot_cos_ptr + rot_base_idx, &cos_vec);
  Load(rot_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(out + store_idx + i) =
        qk_vec[i] * cos_vec[i] + rot_half_vec[i] * sin_vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void RotateQKVec4(const T* qkv_ptr,
                                             const float* rot_cos_ptr,
                                             const float* rot_sin_ptr,
                                             const int load_idx,
                                             const int store_idx,
                                             const int rot_base_idx,
                                             T* out) {
  using VecT = AlignedVector<T, 4>;
  using VecF = AlignedVector<float, 4>;
  auto to_float = [] __device__(T val) -> float {
    return Converter<T>::to_float(val);
  };
  auto from_float = [] __device__(float val) -> T {
    return Converter<T>::from_float(val);
  };

  VecT qk_vec;
  Load(qkv_ptr + load_idx, &qk_vec);
  VecF rot_half_vec = {-to_float(qk_vec[1]),
                       to_float(qk_vec[0]),
                       -to_float(qk_vec[3]),
                       to_float(qk_vec[2])};
  VecF cos_vec, sin_vec;
  Load(rot_cos_ptr + rot_base_idx, &cos_vec);
  Load(rot_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(out + store_idx + i) = from_float(to_float(qk_vec[i]) * cos_vec[i] +
                                        rot_half_vec[i] * sin_vec[i]);
  }
}

template <typename T>
__device__ __forceinline__ void StoreValue(const T* qkv_ptr,
                                           const int load_idx,
                                           const int store_idx,
                                           T* out) {
  using VecT = AlignedVector<T, 4>;
  VecT v_vec;
  Load(qkv_ptr + load_idx, &v_vec);
  Store(v_vec, out + store_idx);
}

template <typename T, typename WeightType>
__global__ void DispatchApplyRopeQKVVec4Kernel(const T* qkv,
                                               const WeightType* rot_cos,
                                               const WeightType* rot_sin,
                                               ApplyRopeQKVParams param,
                                               T* q_out,
                                               T* k_out,
                                               T* v_out) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int head_dim_idx = (blockIdx.z * blockDim.z + threadIdx.z) * 4;
  int rot_idx = token_idx * param.head_dim + head_dim_idx;
  int load_idx, store_idx;

  if (head_idx < param.q_head_num && head_dim_idx < param.head_dim) {  // q
    load_idx = token_idx * param.token_stride +
               (head_idx + param.q_head_offset) * param.head_stride +
               head_dim_idx;
    store_idx =
        token_idx * param.q_stride + head_idx * param.head_dim + head_dim_idx;
    RotateQKVec4(qkv, rot_cos, rot_sin, load_idx, store_idx, rot_idx, q_out);
  }

  if (head_idx < param.kv_head_num && head_dim_idx < param.head_dim) {  // kv
    load_idx = token_idx * param.token_stride +
               (head_idx + param.k_head_offset) * param.head_stride +
               head_dim_idx;
    store_idx =
        token_idx * param.kv_stride + head_idx * param.head_dim + head_dim_idx;
    RotateQKVec4(qkv, rot_cos, rot_sin, load_idx, store_idx, rot_idx, k_out);
    load_idx = token_idx * param.token_stride +
               (head_idx + param.v_head_offset) * param.head_stride +
               head_dim_idx;
    StoreValue(qkv, load_idx, store_idx, v_out);
  }
}

template <paddle::DataType D>
void ApplyRopeQKVKernel(const paddle::Tensor& qkv,
                        const paddle::Tensor& rot_cos,
                        const paddle::Tensor& rot_sin,
                        const int q_head_num,
                        const int kv_head_num,
                        const int head_dim,
                        paddle::Tensor& q_out,
                        paddle::Tensor& k_out,
                        paddle::Tensor& v_out) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int all_num_elements = qkv.numel();
  const int all_num_head = q_head_num + 2 * kv_head_num;
  auto stream = qkv.stream();

  dim3 block_dims(1, 4, 32);
  dim3 grid_dims(all_num_elements / (all_num_head * head_dim),  // token
                 (std::max(q_head_num, kv_head_num) + block_dims.y - 1) /
                     block_dims.y,  // head
                 (head_dim + (block_dims.z * 4) - 1) /
                     (block_dims.z * 4)  // dim: load vec4 at a time
  );

  // printf("grid: (%d, %d, %d)\n", grid_dims.x, grid_dims.y, grid_dims.z);
  // printf("block: (%d, %d, %d)\n", block_dims.x, block_dims.y, block_dims.z);

  ApplyRopeQKVParams param;
  param.head_dim = head_dim;
  param.token_stride = all_num_head * head_dim;
  param.head_stride = head_dim;
  param.q_stride = q_head_num * head_dim;
  param.kv_stride = kv_head_num * head_dim;
  param.q_head_offset = 0;
  param.k_head_offset = q_head_num;
  param.v_head_offset = q_head_num + kv_head_num;
  param.q_head_num = q_head_num;
  param.kv_head_num = kv_head_num;

  if (qkv.dtype() == rot_cos.dtype()) {
    DispatchApplyRopeQKVVec4Kernel<DataType_, DataType_>
        <<<grid_dims, block_dims, 0, stream>>>(
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<const DataType_*>(rot_cos.data<data_t>()),
            reinterpret_cast<const DataType_*>(rot_sin.data<data_t>()),
            param,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(v_out.data<data_t>()));
  } else if (rot_cos.dtype() == paddle::DataType::FLOAT32) {
    DispatchApplyRopeQKVVec4Kernel<DataType_, float>
        <<<grid_dims, block_dims, 0, stream>>>(
            reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
            reinterpret_cast<const float*>(rot_cos.data<float>()),
            reinterpret_cast<const float*>(rot_sin.data<float>()),
            param,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()),
            reinterpret_cast<DataType_*>(v_out.data<data_t>()));
  } else {
    PD_THROW("Unsupported qk dtype and rope dtype.");
  }
}

std::vector<paddle::Tensor> ApplyRopeQKV(const paddle::Tensor& qkv,
                                         const paddle::Tensor& rot_cos,
                                         const paddle::Tensor& rot_sin,
                                         const int q_head_num,
                                         const int kv_head_num,
                                         const int head_dim) {
  auto qkv_shape = qkv.shape();
  auto token_num = qkv_shape[0];
  auto place = qkv.place();
  auto dtype = qkv.dtype();
  common::DDim q_out_shape, kv_out_shape;
  if (rot_cos.shape().size() == 3) {
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
                    rot_cos.shape().back(),
                    "The last dimension of cos mismatches that of q, "
                    "expect %d but got %d",
                    q_out.shape().back(),
                    rot_cos.shape().back());

  switch (dtype) {
    case paddle::DataType::BFLOAT16:
      ApplyRopeQKVKernel<paddle::DataType::BFLOAT16>(qkv,
                                                     rot_cos,
                                                     rot_sin,
                                                     q_head_num,
                                                     kv_head_num,
                                                     head_dim,
                                                     q_out,
                                                     k_out,
                                                     v_out);
      break;
    case paddle::DataType::FLOAT16:
      ApplyRopeQKVKernel<paddle::DataType::FLOAT16>(qkv,
                                                    rot_cos,
                                                    rot_sin,
                                                    q_head_num,
                                                    kv_head_num,
                                                    head_dim,
                                                    q_out,
                                                    k_out,
                                                    v_out);
      break;
    default:
      PD_THROW("Only support qk dtype of BF16 and F16");
  }

  return {q_out, k_out, v_out};
}

std::vector<std::vector<int64_t>> ApplyRopeQKVInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& cos_shape,
    const std::vector<int64_t>& sin_shape) {
  return {qkv_shape, cos_shape, sin_shape};
}

std::vector<paddle::DataType> ApplyRopeQKVInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& cos_dtype,
    const paddle::DataType& sin_dtype) {
  return {qkv_dtype, cos_dtype, sin_dtype};
}

PD_BUILD_OP(apply_rope_qkv)
    .Inputs({"qkv", "rot_cos", "rot_sin"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"q_head_num:int", "kv_head_num:int", "head_dim:int"})
    .SetKernelFn(PD_KERNEL(ApplyRopeQKV))
    .SetInferShapeFn(PD_INFER_SHAPE(ApplyRopeQKVInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ApplyRopeQKVInferDtype));
