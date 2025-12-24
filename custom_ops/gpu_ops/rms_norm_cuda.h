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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once  // NOLINT

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda.h>          // NOLINT
#include <cuda_runtime.h>  // NOLINT
#endif
#include "paddle/extension.h"

#define DEFAULT_THROW(NAME, TYPE)                           \
  default:                                                  \
    do {                                                    \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'"); \
    } while (0);                                            \
    break

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch (TYPEIN) {                                                            \
    case paddle::DataType::FLOAT32: {                                          \
      using scalar_t_in = float;                                               \
      switch (TYPEOUT) {                                                       \
        case paddle::DataType::FLOAT32: {                                      \
          using scalar_t_out = float;                                          \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case paddle::DataType::FLOAT16: {                                      \
          using scalar_t_out = phi::dtype::float16;                            \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case paddle::DataType::BFLOAT16: {                                     \
          using scalar_t_out = phi::dtype::bfloat16;                           \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
          DEFAULT_THROW(NAME, TYPEOUT);                                        \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case paddle::DataType::FLOAT16: {                                          \
      using scalar_t_in = phi::dtype::float16;                                 \
      using scalar_t_out = phi::dtype::float16;                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case paddle::DataType::BFLOAT16: {                                         \
      using scalar_t_in = phi::dtype::bfloat16;                                \
      using scalar_t_out = phi::dtype::bfloat16;                               \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
      DEFAULT_THROW(NAME, TYPEIN);                                             \
  }

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value,
                                           int laneMask,
                                           int width = WARP_SIZE,
                                           unsigned int mask = 0xffffffff) {
  #ifdef PADDLE_WITH_HIP
  return __shfl_xor(value, laneMask, width);
  #else
  return __shfl_xor_sync(mask,value, laneMask, width);
  #endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value,
                                       int srcLane,
                                       int width = WARP_SIZE,
                                       unsigned int mask = 0xffffffff) {
  #ifdef PADDLE_WITH_HIP
  return __shfl(value, srcLane, width);
  #else
  return __shfl_sync(mask, value, srcLane, width);
  #endif
}

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr,
                                   U& mu,       // NOLINT
                                   U& sigma2,   // NOLINT
                                   U& count) {  // NOLINT
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(const U muB,
                                const U sigma2B,
                                const U countB,
                                U& mu,       // NOLINT
                                U& sigma2,   // NOLINT
                                U& count) {  // NOLINT
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename U>
__device__ void cuRMSOnlineSum(const U curr, U& sigma2) {  // NOLINT
  sigma2 = sigma2 + curr * curr;
}

template <typename U>
__device__ void cuChanRMSOnlineSum(const U sigma2B, U& sigma2) {  // NOLINT
  sigma2 = sigma2 + sigma2B;
}


template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T* __restrict__ vals,
                                  const int n1,
                                  const int n2,
                                  const int i1,
                                  U& mu,      // NOLINT
                                  U& sigma2,  // NOLINT
                                  U* buf,
                                  bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    #ifdef PADDLE_WITH_HIP
    for (int l = 0; l <= 5; ++l)
    #else
    for (int l = 0; l <= 4; ++l)
    #endif
    {
      #ifdef PADDLE_WITH_HIP
      int srcLaneB = (threadIdx.x + (1 << l)) & 63;
      #else
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      #endif
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        U muB = WARP_SHFL(mu, srcLaneB);
        U countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;                  // NOLINT
      U* ibuf = (U*)(ubuf + blockDim.y);  // NOLINT
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2 * wrt_y + 1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            U muB = ubuf[2 * threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(const phi::dtype::float16* __restrict__ vals,
                                  const int n1,
                                  const int n2,
                                  const int i1,
                                  float& mu,      // NOLINT
                                  float& sigma2,  // NOLINT
                                  float* buf,
                                  bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);      // NOLINT
  sigma2 = float(0);  // NOLINT
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const auto* lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {  // NOLINT
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2*)(lvals + l + k)));  // NOLINT
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x, mu, sigma2, count);
          cuWelfordOnlineSum(curr.y, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    #ifdef PADDLE_WITH_HIP
    for (int l = 0; l <= 5; ++l)
    #else
    for (int l = 0; l <= 4; ++l)
    #endif
    {
      #ifdef PADDLE_WITH_HIP
      int srcLaneB = (threadIdx.x + (1 << l)) & 63;
      #else
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      #endif
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        float muB = WARP_SHFL(mu, srcLaneB);
        float countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;                  // NOLINT
      float* ibuf = (float*)(ubuf + blockDim.y);  // NOLINT
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y + 1] = sigma2;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            float muB = ubuf[2 * threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / float(n2);  // NOLINT
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);  // NOLINT
    }
  }
}

template <typename U> __device__
U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template <> __device__
float rsqrt(float v) {
  return rsqrtf(v);
}
template <> __device__
double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

}  // namespace

template <typename T, typename U, typename V>
__device__ void cuApplyLayerNorm_(V* __restrict__ output_vals,
                                  U* __restrict__ mean,
                                  U* __restrict__ invvar,
                                  const T* __restrict__ vals,
                                  const int n1,
                                  const int n2,
                                  const U epsilon,
                                  const V* __restrict__ gamma,
                                  const V* __restrict__ beta,
                                  bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, rms_only);
    const T* lvals = vals + i1 * n2;
    V* ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] =
              gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}



template <typename T, typename U, typename V = T>
__global__ void cuApplyRMSNorm(V* __restrict__ output_vals,
                               U* __restrict__ invvar,
                               const T* __restrict__ vals,
                               const int n1,
                               const int n2,
                               const U epsilon,
                               const V* __restrict__ gamma) {
  cuApplyLayerNorm_<T, U, V>(
      output_vals, NULL, invvar, vals, n1, n2, epsilon, gamma, NULL, true);
}


#ifdef PADDLE_WITH_HIP
static hipDeviceProp_t GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(hipGetDevice(&device) == hipSuccess);
  hipDeviceProp_t prop;
  PD_CHECK(hipGetDeviceProperties(&prop, device) == hipSuccess);
  return prop;
}

static hipDeviceProp_t* GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}

#else

static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  return prop;
}

static cudaDeviceProp* GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}
#endif


template <typename T, typename U, typename V = T>
#ifdef PADDLE_WITH_HIP
void HostApplyRMSNorm(V* output,
                      U* invvar,
                      const T* input,
                      int n1,
                      int n2,
                      double epsilon,
                      const V* gamma,
                      hipStream_t stream)
#else
void HostApplyRMSNorm(V* output,
                      U* invvar,
                      const T* input,
                      int n1,
                      int n2,
                      double epsilon,
                      const V* gamma,
                      cudaStream_t stream)
#endif
{
  // auto stream = at::cuda::getCurrentCUDAStream().stream();
  #ifdef PADDLE_WITH_HIP
  const dim3 threads(64, 4, 1);
  #else
  const dim3 threads(32, 4, 1);
  #endif
  // const uint64_t maxGridY =
  // at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  const uint64_t maxGridY = GetDeviceProp()->maxGridSize[1];
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}


static void cuda_rms_norm(const paddle::Tensor& x,
                          const paddle::Tensor& scale,
                          int rows,
                          int cols,
                          float epsilon,
                          paddle::Tensor* y,
                          paddle::Tensor* invvar) {
  DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      x.type(),
      y->type(),
      "cuda_rms_norm_kernel",
      HostApplyRMSNorm(y->data<scalar_t_out>(),
                       invvar->data<float>(),
                       const_cast<scalar_t_in*>(x.data<scalar_t_in>()),
                       rows,
                       cols,
                       epsilon,
                       const_cast<scalar_t_out*>(scale.data<scalar_t_out>()),
                       x.stream()));
}
