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

#include "helper.h"
#include "paddle/extension.h"

__forceinline__ __device__ float tanh_ptx(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float gelu_tanh_func(const float& val) {
  const float cdf =
      0.5f * (1.0f + tanh_ptx((0.7978845608028654f *
                               (val + 0.044715f * val * val * val))));
  return val * cdf;
}

template <typename T>
__global__ void gelu_tanh_kernel(T* __restrict__ out,
                                 const T* __restrict__ input,
                                 const int d) {
  constexpr uint32_t kVecSize = 16 / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * d;
  using vec_t = AlignedVector<T, kVecSize>;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / kVecSize; idx += stride) {
    vec_t x_vec;
    Load(input + offset + idx * kVecSize, &x_vec);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      x_vec[i] = static_cast<T>(gelu_tanh_func(static_cast<float>(x_vec[i])));
    }
    Store(x_vec, out + token_idx * d + idx * kVecSize);
  }

  const int64_t remaining_offset = d - d % (stride * kVecSize);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * kVecSize); idx += stride) {
    float x = static_cast<float>(input[offset + remaining_offset + idx]);
    out[token_idx * d + remaining_offset + idx] =
        static_cast<T>(gelu_tanh_func(x));
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

std::vector<paddle::Tensor> GeluTanh(paddle::Tensor& input) {
  int d = input.dims()[1];
  int64_t num_tokens = input.dims()[0];
  cudaStream_t stream = input.stream();

  paddle::Tensor output =
      GetEmptyTensor(input.dims(), input.dtype(), input.place());

  DISPATCH_FLOAT_FP6_DTYPE(input.dtype(), scalar_t, {
    uint32_t vec_size = 16 / sizeof(scalar_t);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = false;
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config,
                       gelu_tanh_kernel<scalar_t>,
                       output.data<scalar_t>(),
                       input.data<scalar_t>(),
                       d);
  });

  return {output};
}

PD_BUILD_STATIC_OP(gelu_tanh)
    .Inputs({"input"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(GeluTanh));
