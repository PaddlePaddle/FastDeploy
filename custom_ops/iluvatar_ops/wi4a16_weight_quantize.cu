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

// Wi4A16 weight quantization: per-group symmetric int4 with scale = max|w|/7,
// packed two int4 per int8 along the output dimension, matching Python
// fastdeploy.model_executor.ops.iluvatar.utils.wi4a16_weight_quantize_cuda.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "helper.h"

namespace {

__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }

__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ void WriteScale(__half* scales, int idx, float s) {
  scales[idx] = __float2half(s);
}

__device__ __forceinline__ void WriteScale(__nv_bfloat16* scales,
                                           int idx,
                                           float s) {
  scales[idx] = __float2bfloat16_rn(s);
}

template <typename T>
__global__ void Wi4A16QuantizeGroupsKernel(const T* __restrict__ w,
                                           int8_t* __restrict__ q,
                                           T* __restrict__ scales,
                                           int k,
                                           int n,
                                           int group_size,
                                           int num_groups_per_row) {
  const int gid = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid >= group_size) return;

  const int nn = gid / num_groups_per_row;
  const int gj = gid % num_groups_per_row;
  const int kk = gj * group_size + tid;

  extern __shared__ float sdata[];
  const float v = fabsf(ToFloat(w[kk * n + nn]));
  sdata[tid] = v;
  __syncthreads();

  for (int s = group_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  __shared__ float s_scale;
  if (tid == 0) {
    const float max_abs = sdata[0];
    s_scale = max_abs / 7.0f;
  }
  __syncthreads();

  const float scale = s_scale;
  if (tid == 0) {
    WriteScale(scales, gj * n + nn, scale);
  }

  const float wval = ToFloat(w[kk * n + nn]);
  float qf = roundf(wval / scale);
  if (qf > 7.f) qf = 7.f;
  if (qf < -8.f) qf = -8.f;
  q[kk * n + nn] = static_cast<int8_t>(static_cast<int>(qf));
}

__global__ void Wi4A16PackInt4Kernel(const int8_t* __restrict__ q,
                                     int8_t* __restrict__ packed,
                                     int k,
                                     int n) {
  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  const int kk = blockIdx.y;
  const int nhalf = n >> 1;
  if (nn >= nhalf || kk >= k) return;

  const int8_t q0 = q[kk * n + (nn << 1)];
  const int8_t q1 = q[kk * n + (nn << 1) + 1];
  const uint32_t b0 = static_cast<uint32_t>(static_cast<uint8_t>(q0)) & 0xFU;
  const uint32_t b1 = static_cast<uint32_t>(static_cast<uint8_t>(q1)) & 0xFU;
  packed[nn * k + kk] = static_cast<int8_t>(b0 | (b1 << 4));
}

}  // namespace

std::vector<paddle::Tensor> Wi4A16Quantize(const paddle::Tensor& w,
                                           int32_t group_size) {
  PD_CHECK(w.dims().size() == 2,
           "wi4a16_weight_quantize: weight must be 2D [k, n]");
  PD_CHECK(group_size == 128,
           "wi4a16_weight_quantize CUDA: group_size must be 128");
  const int64_t k = w.dims()[0];
  const int64_t n = w.dims()[1];
  PD_CHECK(n % 2 == 0, "wi4a16_weight_quantize: n (dim 1) must be even");
  PD_CHECK(k % group_size == 0,
           "wi4a16_weight_quantize: k must be divisible by group_size");

  PD_CHECK(w.dtype() == paddle::DataType::FLOAT16 ||
               w.dtype() == paddle::DataType::BFLOAT16,
           "wi4a16_weight_quantize: weight dtype must be float16 or bfloat16");
  PD_CHECK(w.is_contiguous(),
           "wi4a16_weight_quantize: weight must be contiguous");

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(w.place()));
  auto stream = static_cast<cudaStream_t>(dev_ctx->stream());

  auto packed = GetEmptyTensor({n / 2, k}, paddle::DataType::INT8, w.place());
  auto scales = GetEmptyTensor({k / group_size, n}, w.dtype(), w.place());
  auto zeros = GetEmptyTensor({k / group_size, n}, w.dtype(), w.place());

  CUDA_CHECK(cudaMemsetAsync(
      zeros.data(),
      0,
      static_cast<size_t>(zeros.numel()) * phi::SizeOf(zeros.dtype()),
      stream));

  auto q_tmp = GetEmptyTensor({k, n}, paddle::DataType::INT8, w.place());
  int8_t* q_ptr = q_tmp.data<int8_t>();
  int8_t* packed_ptr = packed.data<int8_t>();

  const int num_groups_per_row = static_cast<int>(k / group_size);
  const int total_groups = static_cast<int>(n * num_groups_per_row);
  const int threads = group_size;
  const size_t shmem = static_cast<size_t>(group_size) * sizeof(float);

  if (w.dtype() == paddle::DataType::FLOAT16) {
    Wi4A16QuantizeGroupsKernel<__half>
        <<<total_groups, threads, shmem, stream>>>(
            reinterpret_cast<const __half*>(w.data()),
            q_ptr,
            reinterpret_cast<__half*>(scales.data()),
            static_cast<int>(k),
            static_cast<int>(n),
            group_size,
            num_groups_per_row);
  } else {
    Wi4A16QuantizeGroupsKernel<__nv_bfloat16>
        <<<total_groups, threads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(w.data()),
            q_ptr,
            reinterpret_cast<__nv_bfloat16*>(scales.data()),
            static_cast<int>(k),
            static_cast<int>(n),
            group_size,
            num_groups_per_row);
  }

  const int nhalf = static_cast<int>(n >> 1);
  dim3 block(256);
  dim3 grid((nhalf + block.x - 1) / block.x, static_cast<unsigned>(k));
  Wi4A16PackInt4Kernel<<<grid, block, 0, stream>>>(
      q_ptr, packed_ptr, static_cast<int>(k), static_cast<int>(n));

  return {packed, scales, zeros};
}

std::vector<std::vector<int64_t>> Wi4A16QuantizeInferShape(
    const std::vector<int64_t>& w_shape, int32_t group_size) {
  const int64_t k = w_shape[0];
  const int64_t n = w_shape[1];
  const int64_t k_groups = k / group_size;
  return {{n / 2, k}, {k_groups, n}, {k_groups, n}};
}

std::vector<paddle::DataType> Wi4A16QuantizeInferDtype(
    const paddle::DataType& w_dtype, int32_t group_size) {
  return {paddle::DataType::INT8, w_dtype, w_dtype};
}

PD_BUILD_STATIC_OP(wi4a16_weight_quantize_cuda)
    .Inputs({"w"})
    .Outputs({"quant_weight", "scales", "zeros"})
    .Attrs({"group_size: int"})
    .SetKernelFn(PD_KERNEL(Wi4A16Quantize))
    .SetInferShapeFn(PD_INFER_SHAPE(Wi4A16QuantizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Wi4A16QuantizeInferDtype));
