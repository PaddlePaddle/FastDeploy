// Stub for torch's c10/cuda/CUDAException.h to allow BSA compilation under Paddle.
// Inference path never triggers these checks in error state; they only need to
// resolve at compile time.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(EXPR)                                                   \
  do {                                                                         \
    cudaError_t __err = (EXPR);                                                \
    if (__err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(__err));                     \
    }                                                                          \
  } while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA kernel launch error: ") +    \
                               cudaGetErrorString(__err));                     \
    }                                                                          \
  } while (0)
#endif
