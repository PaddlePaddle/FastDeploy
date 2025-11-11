// adapted from:
// https://github.com/flashinfer-ai/flashinfer/blob/0e48aaf941a6b05f6557c9c9f606884f826afedd/include/flashinfer/norm.cuh
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cub/cub.cuh>
#include "cuda.h"
#include "helper.h"

namespace fastdeploy {
namespace norm {

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly
 * shuffle between threads in a warp. \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^
 * delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

template <typename T>
inline __device__ __host__ T div_up(T m, int n) {
  return (m + n - 1) / n;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(T* __restrict__ input,
                                      T* __restrict__ residual,
                                      T* __restrict__ weight,
                                      const uint32_t d,
                                      const uint32_t stride_input,
                                      const uint32_t stride_residual,
                                      float weight_bias,
                                      float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = div_up(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + div_up(num_warps, 4) * 4;
  using vec_t = AlignedVector<T, VEC_SIZE>;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t input_vec;
    Fill(&input_vec, static_cast<T>(0.f));
    vec_t residual_vec;
    Fill(&residual_vec, static_cast<T>(0.f));
    AlignedVector<float, VEC_SIZE> x_vec;
    Fill(&x_vec, 0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      Load(input + bx * stride_input + i * num_threads * VEC_SIZE +
               thread_id * VEC_SIZE,
           &input_vec);
      Load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
               thread_id * VEC_SIZE,
           &residual_vec);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      Store(residual_vec,
            residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                thread_id * VEC_SIZE);
      Store(x_vec, smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t input_vec;
    vec_t weight_vec;
    AlignedVector<float, VEC_SIZE> x_vec;

    Fill(&input_vec, static_cast<T>(0.f));
    Fill(&weight_vec, static_cast<T>(0.f));
    Fill(&x_vec, 0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      Load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE,
           &weight_vec);
      Load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE, &x_vec);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      Store(input_vec,
            input + bx * stride_input + i * num_threads * VEC_SIZE +
                thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
cudaError_t FusedAddRMSNorm(T* input,
                            T* residual,
                            T* weight,
                            uint32_t batch_size,
                            uint32_t d,
                            uint32_t stride_input,
                            uint32_t stride_residual,
                            float eps = 1e-5,
                            bool enable_pdl = false,
                            cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = div_up(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (div_up(num_warps, 4) * 4 + d) * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input,
                  &residual,
                  &weight,
                  &d,
                  &stride_input,
                  &stride_residual,
                  &weight_bias,
                  &eps};

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaLaunchKernelEx(&config,
                       kernel,
                       input,
                       residual,
                       weight,
                       d,
                       stride_input,
                       stride_residual,
                       weight_bias,
                       eps);
  });

  return cudaSuccess;
}

}  // namespace norm
}  // namespace fastdeploy
