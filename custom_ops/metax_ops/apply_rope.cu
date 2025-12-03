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

#define THREADS_PER_BLOCK 128

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

template <typename T>
__device__ void RotateQKVec4(const T* qk_ptr,
                             const T* rot_cos_ptr,
                             const T* rot_sin_ptr,
                             const int head_num,
                             const int base_idx,
                             const int rot_base_idx,
                             T* out) {
  using VecT = AlignedVector<T, 4>;

  VecT qk_vec;
  Load(qk_ptr + base_idx, &qk_vec);
  VecT rot_half_vec = {-qk_vec[1], qk_vec[0], -qk_vec[3], qk_vec[2]};
  VecT cos_vec, sin_vec;
  Load(rot_cos_ptr + rot_base_idx, &cos_vec);
  Load(rot_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(out + base_idx + i) =
        qk_vec[i] * cos_vec[i] + rot_half_vec[i] * sin_vec[i];
  }
}

template <typename T>
__device__ void RotateQKVec4(const T* qk_ptr,
                             const float* rot_cos_ptr,
                             const float* rot_sin_ptr,
                             const int head_num,
                             const int base_idx,
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
  Load(qk_ptr + base_idx, &qk_vec);
  VecF rot_half_vec = {-to_float(qk_vec[1]),
                       to_float(qk_vec[0]),
                       -to_float(qk_vec[3]),
                       to_float(qk_vec[2])};
  VecF cos_vec, sin_vec;
  Load(rot_cos_ptr + rot_base_idx, &cos_vec);
  Load(rot_sin_ptr + rot_base_idx, &sin_vec);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(out + base_idx + i) = from_float(to_float(qk_vec[i]) * cos_vec[i] +
                                       rot_half_vec[i] * sin_vec[i]);
  }
}

// qk and rope have a same type
template <typename T>
__global__ void DispatchApplyRopeVec4Kernel(const T* q,
                                            const T* k,
                                            const T* rot_cos,
                                            const T* rot_sin,
                                            const int q_num_elements,
                                            const int k_num_elements,
                                            const int q_head_num,
                                            const int k_head_num,
                                            const int head_dim,
                                            T* q_out,
                                            T* k_out) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int head_dim_idx = idx % head_dim;

  if (idx < q_num_elements) {
    int rot_idx = idx / (q_head_num * head_dim) * head_dim + head_dim_idx;
    RotateQKVec4(q, rot_cos, rot_sin, q_head_num, idx, rot_idx, q_out);
  }

  if (idx < k_num_elements) {
    int rot_idx = idx / (k_head_num * head_dim) * head_dim + head_dim_idx;
    RotateQKVec4(k, rot_cos, rot_sin, k_head_num, idx, rot_idx, k_out);
  }
}

// rope dtype is float32
template <typename T>
__global__ void DispatchApplyRopeVec4Kernel(const T* q,
                                            const T* k,
                                            const float* rot_cos,
                                            const float* rot_sin,
                                            const int q_num_elements,
                                            const int k_num_elements,
                                            const int q_head_num,
                                            const int k_head_num,
                                            const int head_dim,
                                            T* q_out,
                                            T* k_out) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int head_dim_idx = idx % head_dim;

  if (idx < q_num_elements) {
    int rot_idx = idx / (q_head_num * head_dim) * head_dim + head_dim_idx;
    RotateQKVec4(q, rot_cos, rot_sin, q_head_num, idx, rot_idx, q_out);
  }

  if (idx < k_num_elements) {
    int rot_idx = idx / (k_head_num * head_dim) * head_dim + head_dim_idx;
    RotateQKVec4(k, rot_cos, rot_sin, k_head_num, idx, rot_idx, k_out);
  }
}

template <paddle::DataType D>
void ApplyRopeKernel(const paddle::Tensor& q,
                     const paddle::Tensor& k,
                     const paddle::Tensor& rot_cos,
                     const paddle::Tensor& rot_sin,
                     paddle::Tensor& q_out,
                     paddle::Tensor& k_out) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto q_num_elements = q.numel();
  const auto k_num_elements = k.numel();
  const auto q_shape = q.shape();
  const auto k_shape = k.shape();
  const auto dims = q_shape.size();
  const auto q_head_num = q_shape[dims - 2];
  const auto k_head_num = k_shape[dims - 2];
  const auto head_dim = q_shape.back();
  int block_num =
      (std::max(q_num_elements, k_num_elements) + (THREADS_PER_BLOCK * 4) - 1) /
      (THREADS_PER_BLOCK * 4);
  auto stream = q.stream();

  if (q.dtype() == rot_cos.dtype()) {
    DispatchApplyRopeVec4Kernel<DataType_>
        <<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const DataType_*>(q.data<data_t>()),
            reinterpret_cast<const DataType_*>(k.data<data_t>()),
            reinterpret_cast<const DataType_*>(rot_cos.data<data_t>()),
            reinterpret_cast<const DataType_*>(rot_sin.data<data_t>()),
            q_num_elements,
            k_num_elements,
            q_head_num,
            k_head_num,
            head_dim,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()));
  } else if (rot_cos.dtype() == paddle::DataType::FLOAT32) {
    DispatchApplyRopeVec4Kernel<DataType_>
        <<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const DataType_*>(q.data<data_t>()),
            reinterpret_cast<const DataType_*>(k.data<data_t>()),
            reinterpret_cast<const float*>(rot_cos.data<float>()),
            reinterpret_cast<const float*>(rot_sin.data<float>()),
            q_num_elements,
            k_num_elements,
            q_head_num,
            k_head_num,
            head_dim,
            reinterpret_cast<DataType_*>(q_out.data<data_t>()),
            reinterpret_cast<DataType_*>(k_out.data<data_t>()));
  } else {
    PD_THROW("Unsupported qk dtype and rope dtype.");
  }
}

std::vector<paddle::Tensor> ApplyRope(const paddle::Tensor& q,
                                      const paddle::Tensor& k,
                                      const paddle::Tensor& rot_cos,
                                      const paddle::Tensor& rot_sin) {
  auto q_shape = q.shape();
  auto cos_shape = rot_cos.shape();

  auto q_out = paddle::empty_like(q);
  auto k_out = paddle::empty_like(k);

  if (q.numel() == 0 || k.numel() == 0) {
    return {q_out, k_out};
  }

  PADDLE_ENFORCE_EQ(
      q_shape.back() % 2,
      0,
      "The last dimension (head_dim) of qk must be an even number "
      "for RoPE, but got %d",
      q_shape.back());
  PADDLE_ENFORCE_EQ(q_shape.size(),
                    cos_shape.size(),
                    "The shape size of cos mismatches the shape size of q, "
                    "expect %d but got %d",
                    q_shape.size(),
                    cos_shape.size());
  PADDLE_ENFORCE_EQ(q_shape.back(),
                    cos_shape.back(),
                    "The shape.back() of cos mismatches the shape.back() of q, "
                    "expect %d but got %d",
                    q_shape.back(),
                    cos_shape.back());

  auto input_type = q.dtype();
  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      ApplyRopeKernel<paddle::DataType::BFLOAT16>(
          q, k, rot_cos, rot_sin, q_out, k_out);
      break;
    case paddle::DataType::FLOAT16:
      ApplyRopeKernel<paddle::DataType::FLOAT16>(
          q, k, rot_cos, rot_sin, q_out, k_out);
      break;
    default:
      PD_THROW("Only support qk dtype of BF16 and F16");
  }

  return {q_out, k_out};
}

std::vector<std::vector<int64_t>> ApplyRopeInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& cos_shape,
    const std::vector<int64_t>& sin_shape) {
  return {q_shape, k_shape, cos_shape, sin_shape};
}

std::vector<paddle::DataType> ApplyRopeInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& cos_dtype,
    const paddle::DataType& sin_dtype) {
  return {q_dtype, k_dtype, cos_dtype, sin_dtype};
}

PD_BUILD_OP(apply_rope)
    .Inputs({"q", "k", "rot_cos", "rot_sin"})
    .Outputs({"q_out", "k_out"})
    .SetKernelFn(PD_KERNEL(ApplyRope))
    .SetInferShapeFn(PD_INFER_SHAPE(ApplyRopeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ApplyRopeInferDtype));
